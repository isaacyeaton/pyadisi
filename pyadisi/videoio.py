from __future__ import division

import numpy as np

import re
import subprocess as sp
import sys
import os

import time


def try_ffmpeg(FFMPEG_BINARY):
    try:
        proc = sp.Popen([FFMPEG_BINARY],
                        stdout=sp.PIPE,
                        stderr=sp.PIPE)
        proc.wait()
    except:
        return False
    else:
        return True


FFMPEG_BINARY_SUGGESTIONS = ['ffmpeg', 'ffmpeg.exe']

FFMPEG_BINARY = None
for name in FFMPEG_BINARY_SUGGESTIONS:
    if try_ffmpeg(name):
        FFMPEG_BINARY = name
        break

PIX_FMT = {'rgb24': 3,
           'rgba': 4}

def kill_process(proc):
    """Kill a running process and close stdin, stdout, and stderr

    Parameters
    ----------
    proc : subprocess.Popen
        Open subprocess

    Returns
    -------
    None
    """

    proc.terminate()
    for std in proc.stdin, proc.stdout, proc.stderr:
        std.close()


def open_process(cmd):
    """Open a process to talk with ffmpeg.

    Parameters
    ----------
    cmd : list
        Command in the form of a list. See subprocess documentation

    Returns
    -------
    proc : subprocess.Popen
        Open subprocess
    """

    proc = sp.Popen(cmd, stdin=sp.PIPE,
                     stdout=sp.PIPE,
                     stderr=sp.PIPE)
    return proc


def video_info(filename, pix_fmt):
    """Load information about the video.

    Parameters
    ----------
    filename : str
        Filename of the video to open (.mp4, .mov, .avi, etc.)
    pix_fmt : str
        Pixel format for ffpmeg. 'rgb24 is a good bet.

    Returns
    -------
    vid_info : list
        Output from proc.stderr.readlines() of ffmpeg that contains information
        about the video and ffmpeg options.
    """

    cmd = [FFMPEG_BINARY, '-i', filename,
                    '-f', 'image2pipe',
                    '-pix_fmt', pix_fmt,
                    '-vcodec', 'rawvideo',
                    '-f', 'null',
                    os.devnull]

    proc = open_process(cmd)
    vid_info = proc.stderr.readlines()
    kill_process(proc)

    return vid_info


def video_parse_info(vid_info):
    """Parse the stdout from ffmpeg to get frame size and number
    of frames in the video.

    Parameters
    ----------
    vid_info : list
        Output from video_info

    Returns
    -------
    nframes : int
        Number of frames
    width : int
        Video width in pixels
    height : int
        Video height in pixels
    """

    # get the output lines that describe the video
    line = [l for l in vid_info if ' Video: ' in l][0]

    # get the size, of the form 460x320 (w x h)
    match = re.search(" [0-9]*x[0-9]*(,| )", line)
    width, height = map(int, line[match.start():match.end()-1].split('x'))

    # get the number of frames (should be a better way to get this...)
    nframes = int(vid_info[-2].split('\r')[-2].split()[1])

    return nframes, width, height


def video_read(filename, pix_fmt):
    """Open the video so we can read binary data from it.

    Parameters
    ----------
    filename : str
        Filename of the video to open (.mp4, .mov, .avi, etc.)
    pix_fmt : str
        Pixel format for ffpmeg. 'rgb24 is a good bet.

    Returns
    -------
    proc : subprocess.Popen
        Process connected to ffmpeg to read binary output.
    """

    cmd = [FFMPEG_BINARY, '-i', filename,
                    '-f', 'image2pipe',
                    '-pix_fmt', pix_fmt,
                    '-vcodec', 'rawvideo', '-']

    proc = open_process(cmd)
    return proc


def write_npy_header(fp, nframes, width, height, depth, dtype):
    """Write a numpy header so we can use np.load
    to read it back in.

    Parameters
    ----------
    fp : file
        Open binary file.
    nframes : int
        Number of frames from the video.
    width : int
        Width of the video in pixels.
    height : int
        Height of the video in pixels.
    depth : int
        Color depth of the video.
    dtype : type
        Datatype of the video

    Returns:
    None
    """

    import struct

    # Get the dict normally used for a .npy file header, but
    # change what the shape will be, as we are going to
    # continually write to that file.
    tmp = np.zeros(0, dtype=dtype)
    header_info = np.lib.format.header_data_from_array_1_0(tmp)
    header_info['shape'] = (nframes, height, width, depth)  # note: we want height, width

    # Do the rest of this because np.lib.format.write_array_header_1_0(fp, header_info)
    # was not writing out the magic '\x93NUMPY\x01\x00' to the file; it only started with
    # the header_len_str. Therefore, I am writing this stuff out myself, assuming
    # version=(1, 0).
    # See: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py#L270
    header = ["{"]
    for key, value in sorted(header_info.items()):
        # Need to use repr here, since we eval these when reading
        header.append("'%s': %s, " % (key, repr(value)))
    header.append("}")
    header = "".join(header)

    current_header_len = np.lib.format.MAGIC_LEN + 2 + len(header) + 1  # 1 for the newline
    topad = 16 - (current_header_len % 16)
    header = np.compat.asbytes(header + ' '*topad + '\n')

    header_len_str = struct.pack('<H', len(header))

    version = (1, 0)

    fp.write(np.lib.format.magic(*version))
    fp.write(header_len_str)
    fp.write(header)


def video_to_npy(fp, proc, nreadwrite, nframes, width, height, depth, dtype):
    """Read from the opened video process and write the data to
    an open binary file. We do this reading only a set number of frames
    each time and writing them, to keep ram use low.

    Parameters
    ----------
    fp : file
        Open binary file that has numpy header written to it.
    proc : subprocess.Popen
        Output from video_read.
    nreadwrite : int
        Number of frames to read before out to disk. This many
        frames will be loaded into memory at any given time.
    nframes : int
        Number of frames from the video.
    width : int
        Width of the video in pixels.
    height : int
        Height of the video in pixels.
    depth : int
        Color depth of the video.
    dtype : type
        Datatype of the video
    """

    BASEREAD = width * height * depth

    nleft = nframes
    if nreadwrite > nframes:
        nreadwrite = nframes

    while True:
        arr = np.fromstring(proc.stdout.read(nreadwrite * BASEREAD), dtype=dtype)
        arr.tofile(fp)  # turns out don't need to do reshaping
        #https://github.com/soft-matter/pims/blob/master/pims/ffmpeg_reader.py
        #arr.reshape(nread, height, width, depth).tofile(fp)

        #print('nread: {0} \t nleft: {1}'.format(nreadwrite, nleft))
        to_read = nleft - nreadwrite

        if to_read < 0:
            nreadwrite = np.abs(nleft)
        elif to_read == 0:
            break

        nleft -= nreadwrite
    #print('\nfinished!\nnreadwrite: {0} \t nleft: {1}'.format(nreadwrite, nleft))

    return


def load_video(filename, mmap_mode='c', pix_fmt='rgb24', nreadwrite=100, timing=False):
    """Load in a video file as a numpy array (if we have decoded the video before),
    or make this array (and a binary file on disk to store it).

    Note: If the height and width are swapped, the following will swap the axes:
    >>> dat = dat.swapaxes(1, 2)

    Parameters
    ----------
    filename : str
        Filename of the video to open (.mp4, .mov, .avi, etc.)
    mmap_mode : {r+', 'r', 'w+', 'c'}, default is 'c'
        How to memory map the file. Default is copy-on-write.
        See np.memmap and np.load for more details.
    pix_fmt : 'rgb24' or 'rgba', default is 'rgb24'
        Pixel format of the video (passed to ffmpeg).
    nreadwrite : int, default is 100
        Number of frames to hold in memory at any given time
        when making the numpy binary file.
    timing : bool, default is False
        Printing timing information about saving the file
        to disk.

    Returns
    -------
    dat : ndarray
        Array of the video. Shape is (nframes, height, width, depth),
        where depth is the color depth for the format.

    Notes
    -----
    We take the approach of the pims project (github.com/soft-matter/pims) and use
    ffmpeg to read in raw byte stream(?) from ffmpeg and write that out to
    disk as a binary file. This allows the entire video sequence to be analyzed
    at once (as a 4D array) and easy indexing for different frames.
    Memmapping prevents the entire file from being read in at once.

    We differ by using .npy binary files, which are
    platform independent and have a defined shape. This way we don't rely on things
    being platform dependent or a plain text file on how to interpret the binary file.

    Information on this format and why it is preferrable can be found here:
    github.com/numpy/numpy/blob/master/doc/neps/npy-format.rst
    Some additional information on avoiding plain binary can be found in the
    tofile method of np.array.
    """

    savename = filename + '.npy'

    if os.path.isfile(savename):
        dat = np.load(savename, mmap_mode=mmap_mode)
    else:
        if timing:
            now = time.time()

        depth = PIX_FMT[pix_fmt]
        dtype = np.uint8

        vid_info = video_info(filename, pix_fmt)
        nframes, width, height = video_parse_info(vid_info)

        with open(savename, 'wb') as fp:
            proc = video_read(filename, pix_fmt)
            write_npy_header(fp, nframes, width, height, depth, dtype)
            video_to_npy(fp, proc, nreadwrite, nframes, width, height, depth, dtype)
            kill_process(proc)

        dat = np.load(savename, mmap_mode=mmap_mode)

        if timing:
            print('Elapsed time: {0:.3f} sec.'.format(time.time() - now))

    return dat

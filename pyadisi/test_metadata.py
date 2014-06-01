from __future__ import division

import metadata

def test_fastec():
    fname = '../test_data/metadata.txt'

    d = metadata.fastec(fname)

    assert d.has_key('Image') == True
    assert d.has_key('Camera') == True
    assert d.has_key('Record') == True
    assert d.has_key('Normalization') == True
    
    assert d['Image']['roi_x'] == 0
    assert d['Image']['roi_y'] == 0
    assert d['Image']['width'] == 1280

    assert d['Normalization']['green_matrix'] == [-131, 381, 5]
    

    f = '''
    [Image]
        roi_x=0
        roi_y=0
        width=1280
        height=1024
        bit_mode=full 10
        frame_count=2193
        trigger_frame=4883
        start_frame=412
        end_frame=2604
        time_stamp=14:148:09:12:13.125242

    [Camera]
        make=FASTEC
        model=TS3100SC8256
        fpga_rev=0x000100ba
        software_version=1.6.47
        mac_address=a4:1b:c0:00:00:1a
        camera_name=FASTEC-TS3-1A
        sensor_type=C31L

    [Record]
        fps=500
        shutter_speed=500
        multi_slope=0:0
        trigger_settings=0
        sync_in=0x0
        sync_out=0x0

    [Normalization]
        red_balance=256
        blue_balance=558
        green_balance=307
        brightness=100
        contrast=100
        gamma=100
        sensor_gain=100
        red_gain=0
        green_gain=0
        blue_gain=0
        red_matrix=[182,159,-82]
        blue_matrix=[26,-392,673]
        green_matrix=[-131,381,5]
        raw=0
        codec=TIFF'''

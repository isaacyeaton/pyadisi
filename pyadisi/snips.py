"""Useful snippets found on the internet.
"""

# https://github.com/qutip/qutip/blob/master/qutip/ipynbtools.py
from IPython.parallel import Client
from IPython.display import HTML, Javascript, display

import matplotlib.pyplot as plt
from matplotlib import animation

def plot_animation(plot_setup_func, plot_func, result, name="movie",
                   verbose=False):
    """
    Create an animated plot of a Odedata object, as returned by one of
    the qutip evolution solvers.

    .. note :: experimental
    """


    fig, axes = plot_setup_func(result)

    def update(n): 
        plot_func(result, n, fig=fig, axes=axes)

    anim = animation.FuncAnimation(fig, update, frames=len(result.times), blit=True)

    anim.save(name + '.mp4', fps=10, writer='ffmpeg', codec='libx264')

    plt.close(fig)
    
    if verbose:
        print("Created %s.m4v" % name)
    
    video = open(name + '.mp4', "rb").read()
    video_encoded = video.encode("base64")
    video_tag = '<video controls src="data:video/x-m4v;base64,{0}">'.format(video_encoded)
    return HTML(video_tag)


from __future__ import absolute_import, division, print_function
from six.moves import range, zip
import six

# built-in
import os

# 3rd party
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# local
from . import utils
from identification import tools as st

# Globals
MAJOR_LW = 2.5
MINOR_LW = 1.5
MAX_ROWS = 10


def plot_ecg(ts=None,
             raw=None,
             filtered=None,
             rpeaks=None,
             templates_ts=None,
             templates=None,
             heart_rate_ts=None,
             heart_rate=None,
             path=None,
             show=False):
    """Create a summary plot from the output of signals.ecg.ecg.

    Parameters
    ----------
    ts : array
        Signal time axis reference (seconds).
    raw : array
        Raw ECG signal.
    filtered : array
        Filtered ECG signal.
    rpeaks : array
        R-peak location indices.
    templates_ts : array
        Templates time axis reference (seconds).
    templates : array
        Extracted heartbeat templates.
    heart_rate_ts : array
        Heart rate time axis reference (seconds).
    heart_rate : array
        Instantaneous heart rate (bpm).
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    fig = plt.figure()
    fig.suptitle('ECG Summary')
    gs = gridspec.GridSpec(6,1)


    # raw signal
    ax1 = fig.add_subplot(gs[:2, 0])
    ax1.set_title('Raw Signal')
    ax1.plot(ts, raw, linewidth=MINOR_LW, label='Raw')

    ax1.set_ylabel('Amplitude')
    #ax1.set_xlabel('Time (s)')
    ax1.legend()
    ax1.grid()


    # filtered signal with rpeaks
    ax2 = fig.add_subplot(gs[2:4, 0], sharex=ax1)
    ax2.set_title('R peaks')
    ymin = np.min(filtered)
    ymax = np.max(filtered)
    alpha = 0.1 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha

    ax2.plot(ts, filtered, linewidth=MINOR_LW, color='m',label='R-peaks')
    # ax2.vlines(ts[rpeaks], ymin, ymax,
    #            color='m',
    #            linewidth=MINOR_LW,
    #            label='R-peaks')

    ax2.set_ylabel('Amplitude')
    #ax2.set_xlabel('Time (s)')
    ax2.legend()
    ax2.grid()

    # heart rate
    ax3 = fig.add_subplot(gs[4:, 0], sharex=ax1)

    ax3.plot(heart_rate_ts, heart_rate, linewidth=MINOR_LW, label='Heart Rate')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Heart Rate (bpm)')
    ax3.legend()
    ax3.grid()

    # # templates
    # ax4 = fig.add_subplot(gs[1:5, 1])
    #
    # ax4.plot(templates_ts, templates.T, 'm', linewidth=MINOR_LW, alpha=0.7)
    #
    # ax4.set_xlabel('Time (s)')
    # ax4.set_ylabel('Amplitude')
    # ax4.set_title('Templates')
    # ax4.grid()

    # make layout tight
    #gs.tight_layout(fig)



    # save to file
    if path is not None:
        path = utils.normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'

        fig.savefig(path, dpi=200, bbox_inches='tight')

    # show
    if show:
        plt.show()
    else:
        # close
        plt.close(fig)

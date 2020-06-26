"""
=========================================================
Plotting Library
=========================================================
"""

import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_figure(figure,name='Default',caption='Default caption.',experiment_name='',dirname='report'):
    """
    Saves a figure.
    """
    if name:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        fname = dirname+os.sep+name+'.pdf'
        pp = PdfPages(fname)

    if fname is not None:
        pp.savefig(figure)
        pp.close()
        plt.close()




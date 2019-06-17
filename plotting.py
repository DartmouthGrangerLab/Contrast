"""
=========================================================
Plotting Library
=========================================================
"""

import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_figure(figure,name='Default',caption='Default caption.',experiment_name=''):
    """
    Saves a figure.
    """
    if name:
        dirname = 'report'+os.sep+experiment_name+'-images'
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        fname = dirname+os.sep+name+'.pdf'
        pp = PdfPages(fname)

    if fname is not None:
        pp.savefig(figure)
        pp.close()
        plt.close()

    result = ('\\begin{figure}[H]\n'
                #'\\centering\n' + 
                '\\begin{center}\n'
                '\\includegraphics[width=0.8\\textwidth]{'+experiment_name+'-images'+os.sep+name+'.pdf}\n' 
                '\\end{center}\n' 
                '\\caption{\\label{fig:'+name+'}'+caption+'}\n' 
                '\\end{figure}\n')
    return result



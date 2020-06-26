#!/usr/bin/env python3
"""
=========================================================
HerzogReport 2012 Report
=========================================================
"""
from builtins import object
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm,sem
import pandas as pd
import os

import sys
from Contrast.model.model import Model

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from Contrast.model.plotting import plot_figure

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from Contrast.model.plotting import plot_figure
from scipy import stats
from Contrast.model.report import Report,main,fit

class HerzogReport(Report):
    """Documentation for Report

    More documentation

    :Parameters:
        visible : **True** or False
            documentation on parameter
        newPos : **None** or [x,y]
            documentation on parameter
    """

    def __init__(self,
                 filenames=[],
                 only_most_recent=False,
                 contrast_results=0,
                 dirname=''):
        super(HerzogReport, self).__init__(filenames=filenames,only_most_recent=only_most_recent,contrast_results=contrast_results,dirname=dirname)
        self.plot_x_scale_factor = 1.0
        self.plot_x_scale_addition = 0.0

        self.ind_params = ['num_flank','offset']
        self.dep_params = ['contrast']
        self.report_col_params = ['num_flank']
        self.report_idx_params = ['offset']
        self.report_val_params = ['contrast']
        
    def _generate(self):
        plot_name = ['a','b','c','d']
        result_df = self.result_df.dropna(axis=0,subset=['trials.thisN'])
        target_contrast = np.unique(result_df['model.target_contrast'])[0]
        est_max = np.unique(result_df['model.est_max'])[0]
        upper_limit=np.unique(result_df['model.upper_limit'])[0]
        lower_limit=np.unique(result_df['model.lower_limit'])[0]
        decision_sigma = np.unique(result_df['model.decision_sigma'])[0]
        decision_K = np.unique(result_df['model.decision_K'])[0]

        for i,df in enumerate(self.df_decision):
            self.plot_decision_func(df,title='Decision Function',xlabel='Contrast',ylabel='Proportion Correct Response',columns_title='Num Flanks',columns_marker_color=['m','g','b','r','c'],plot_columns_labels=['0','1','2','3','4'],fname='decision-'+self.experiment_name+'-Figure-1'+plot_name[i],decision_prob_label='decision_prob',target_contrast=target_contrast,est_max=est_max,upper_limit=upper_limit,lower_limit=lower_limit,sigma=decision_sigma,K=decision_K)

        fname_label = ['a','b','c','d']
        plot_y_margin = 1.1
        plot_y_max = np.max(np.array([np.max(x['contrast'].values) for x in self.df_decision]))*plot_y_margin
        for i,df in enumerate(self.df_decision):
            contrast_results = pd.DataFrame(df['contrast'])
            contrast_results = pd.pivot_table(pd.DataFrame(contrast_results),columns=['num_flank'],aggfunc=np.mean).T

            self.plot_data(contrast_results,fname='contrast-Herzog-2012-Figure-1'+fname_label[i],title='Contrast vs. Num Flanks for '+self.experiment_name,plot_min=0,plot_max=plot_y_max,xlabel='Num Flanks',ylabel='Contrast')

            decision_prob_results = pd.DataFrame(df['decision_prob'])
            decision_prob_results = pd.pivot_table(pd.DataFrame(decision_prob_results),columns=['num_flank'],aggfunc=np.mean).T
            self.plot_data(fit(decision_prob_results),fname='Herzog-2012-Figure-1'+fname_label[i],title='Threshold elevation vs. Num Flanks for '+self.experiment_name,plot_min=0,plot_max=1,xlabel='Num Flanks',ylabel='Threshold elevation')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(self.result_df[['num_flank','contrast','decision_prob','level','offset']])

          
# mark functions to profile with @profile
# profile with: kernprof -l -v framework.py
if __name__ == "__main__":
    main(reportclass=HerzogReport,filenames=sys.argv[1:],only_most_recent=False,dirname=os.getcwd())



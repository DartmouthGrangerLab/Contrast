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

class KahnemanReport(Report):
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
        super(KahnemanReport, self).__init__(filenames=filenames,only_most_recent=only_most_recent,contrast_results=contrast_results,dirname=dirname)

        self.plot_x_scale_factor = 1.0
        self.plot_x_scale_addition = 0.0
        self.ind_params = ['flank_distance','offset']
        self.dep_params = ['contrast']
        self.report_col_params = ['flank_distance']
        self.report_idx_params = ['offset']
        self.report_val_params = ['contrast']
        
    def plot_figures(self,df_resp,fname='',xlabel='',ylabel='',plot_min=0,plot_max=1):
        exp_title='exp_title'
        title = ylabel+' vs. '+ xlabel + ' for '+self.experiment_name

        self.plot_data(df_resp,fname=fname,title=title,xlabel=xlabel,ylabel=ylabel,plot_min=plot_min,plot_max=plot_max) # df_err=df_err[0.5].iloc[:,0]

    def _generate(self):
        result_df = self.result_df.dropna(axis=0,subset=['trials.thisN'])
        target_contrast = np.unique(result_df['model.target_contrast'])[0]
        est_max = np.unique(result_df['model.est_max'])[0]
        upper_limit=np.unique(result_df['model.upper_limit'])[0]
        lower_limit=np.unique(result_df['model.lower_limit'])[0]
        decision_sigma = np.unique(result_df['model.decision_sigma'])[0]
        decision_K = np.unique(result_df['model.decision_K'])[0]

        self.df_decision[0].index = self.df_decision[0].index.set_levels(self.df_decision[0].index.levels[0]*60.0,level=0) 

        self.plot_decision_func(self.df_decision[0][1:],
                                title='Decision Function',
                                xlabel='Contrast',
                                ylabel='Proportion Correct Response',
                                columns_title='Flank Dist.',
                                plot_columns_labels= ['{:3.4f}'.format(x) for x in self.df_decision[0].index.levels[1].values.tolist()],
                                fname='decision-'+self.experiment_name+'-Figure-1',
                                decision_prob_label='decision_prob',
                                target_contrast = target_contrast,
                                target_value=self.df_decision[0]['decision_prob'][0],
                                index_marker_shape=['o','v','^','s','p','*','h','H','+','x','D','d','|','_'],
                                est_max=est_max,
                                upper_limit=upper_limit,
                                lower_limit=lower_limit,
                                sigma=decision_sigma,
                                show_legend=True,
                                legend_loc=4,
                                transpose_df=False,
                                K=decision_K)

        plot_y_margin = 1.1
        plot_y_max = np.max(np.array([np.max(x['contrast'].values) for x in self.df_decision]))*plot_y_margin
        for i,df in enumerate(self.df_decision):
            plot_y_margin = 1.1
            plot_y_max = np.max(df['contrast'].values)*plot_y_margin
            plot_y_min = np.min(df['contrast'].values)*plot_y_margin
            contrast_results = pd.DataFrame(df['contrast'])
            contrast_results = pd.pivot_table(pd.DataFrame(contrast_results),columns=['flank_distance'],aggfunc=np.mean).T

            self.plot_data2(contrast_results,fname='contrast-Kahneman-2012-Figure-1',title='Contrast vs. Flank distance for '+self.experiment_name,plot_min=plot_y_min,plot_max=plot_y_max,xlabel='Flank Distance',ylabel='Contrast')

            decision_prob_results = pd.DataFrame(df['decision_prob'])
            decision_prob_results = pd.pivot_table(pd.DataFrame(decision_prob_results),columns=['flank_distance'],aggfunc=np.mean).T

            self.plot_data2(decision_prob_results*100,fname='Kahneman-2012-Figure-1',title='Percent Correct vs. Flank distance for '+self.experiment_name,plot_min=0,plot_max=100,xlabel='Flank Distance',ylabel='Percent Correct',index_marker_shape=['o','v','^','s','p','*','h','H','+','x','D','d','|','_'],x_scale_factor=1.0)
        print(self.result_df[['flank_distance','contrast','decision_prob']])
          
# mark functions to profile with @profile
# profile with: kernprof -l -v framework.py
if __name__ == "__main__":
    main(reportclass=KahnemanReport,filenames=sys.argv[1:],only_most_recent=False,dirname=os.getcwd())



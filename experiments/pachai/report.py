#!/usr/bin/env python3
"""
=========================================================
PachaiReport Report
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
from Contrast.model.report import Report,main

class PachaiReport(Report):
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
        super(PachaiReport, self).__init__(filenames=filenames,only_most_recent=only_most_recent,contrast_results=contrast_results,dirname=dirname)
        
        self.plot_x_scale_factor = 1.0
        self.plot_x_scale_addition = 0.0
        self.ind_params = ['num_flank','gap','flank_distance']
        self.dep_params = ['contrast']
        self.report_col_params = ['num_flank']
        self.report_idx_params = ['flank_distance','gap']
        self.report_val_params = ['contrast']
        self.exp_results = []

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
            num_flank_levels = np.unique(df.T.columns.get_level_values('num_flank'))
            for ii,num_flank in enumerate(num_flank_levels):
                decision_K = np.unique(result_df['model.decision_K'])[ii]
                est_max = np.unique(result_df['model.est_max'])[ii]

                df_contrast = pd.DataFrame(df['contrast'][num_flank])
                df_contrast = pd.pivot_table(df_contrast,columns=['gap'],index=['flank_distance'])
                df_resp = df.T[num_flank].T
                df_contrast = None

                self.plot_decision_func(df_resp,
                                        df_contrast=df_contrast,
                                        title='Decision Function',
                                        xlabel='Contrast',
                                        ylabel='Proportion Correct Response',
                                        #columns_title='Flank dist.',
                                        columns_title='# Flanks / Gap',
                                        columns_marker_color=['m','g','b','r','c'],
                                        plot_columns_labels=[str(num_flank)+' flank, no gap',str(num_flank)+' flank, gap'],
                                        fname='decision-'+str(num_flank)+'-'+self.experiment_name+'-Figure-'+plot_name[i],
                                        index_marker_shape=['o','v','^','s','p','*','h','H','+','x','D','d','|','_'],
                                        decision_prob_label='decision_prob',
                                        target_contrast=target_contrast,
                                        est_max=est_max,
                                        upper_limit=upper_limit,
                                        lower_limit=lower_limit,
                                        sigma=decision_sigma,
                                        K=decision_K,
                                        compute_error=False,
                                        show_legend=True,
                                        use_target_contrast=False,
                                        transpose_df=False)
       
                xlabel='Flank ring radius / eccentricity'
                ylabel='Contrast'
                title = ylabel+' vs. '+ xlabel + ' for '+self.experiment_name

                contrast_results = pd.pivot_table(pd.DataFrame(df_resp['contrast']),columns=['gap'],index=['flank_distance'])
                #import pdb; pdb.set_trace()
                contrast_results.columns = contrast_results.columns.droplevel() # remove extraneous outer level
                self.plot_data2(contrast_results,fname='contrast-'+str(num_flank)+'-'+'Pachai-Figure-1',plot_max=np.max(np.array(contrast_results.values)),xlabel=xlabel,ylabel=ylabel,title=title,columns_title='# Flanks / Gap',plot_columns_labels=[str(num_flank)+' flank, no gap',str(num_flank)+' flank, gap'],x_scale_factor=0.1)

                exp_results = pd.pivot_table(pd.DataFrame(1-df_resp['decision_prob']),columns=['gap'],index=['flank_distance'])
                exp_results.columns = exp_results.columns.droplevel() # remove
                print('Number of flanks: '+str(num_flank))
                print(exp_results*100)

                xlabel='Flank ring radius / eccentricity'
                ylabel='Percent Error'
                title = ylabel+' vs. '+ xlabel + ' for '+self.experiment_name
                self.plot_data2(exp_results,fname='Pachai-'+str(num_flank)+'-'+'Figure-1',plot_min=0,plot_max=1,xlabel=xlabel,ylabel=ylabel,title=title,columns_title='# Flanks / Gap',plot_columns_labels=[str(num_flank)+' flank, no gap',str(num_flank)+' flank, gap'],x_scale_factor=0.1)
                exp_results = pd.concat([exp_results],names=['num_flank'],keys=[num_flank],axis=1)
                self.exp_results.append(exp_results)
                print(df)

        
        df_resp = pd.concat(self.exp_results,axis=1)*100

        print('Results:')
        print(df_resp)
        self.plot_data_as_barplot2(df_resp,fname='plot-'+self.experiment_name+'-barplot',title='Perceptual error vs. Number of flankers for '+self.experiment_name,xlabel='Number of flankers',ylabel='Perceptual error')


# mark functions to profile with @profile
# profile with: kernprof -l -v framework.py
if __name__ == "__main__":
    main(reportclass=PachaiReport,filenames=sys.argv[1:],only_most_recent=False,dirname=os.getcwd())

#!/usr/bin/env python
"""
=========================================================
Report
=========================================================
"""
from builtins import object
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm,sem
import pandas as pd
import os

import sys
from Contrast.model.model import decision_func

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import stats
from Contrast.model.plotting import plot_figure

class Report(object):
    """Documentation for Report
    """

    def __init__(self,
                 filenames=[],
                 only_most_recent=False,
                 contrast_results=0,
                 dirname=''):
        self.dirname = dirname
        self.filenames = filenames
        self.contrast_results = contrast_results
        self.plot_x_scale_factor = 1.0
        self.plot_x_scale_addition = 0.0
        self.report_col_params = ['num_flank']
        self.report_idx_params = ['model_offset']
        self.report_val_params = ['contrast']

        fnames = []
        for filename in filenames:
            #if filename.startswith(subject) and filename.endswith('.csv'):
            if filename.endswith('.csv'):
                fnames.append(filename)
                print('Loading: '+ filename)
                if only_most_recent:
                    break

        self.result_df = pd.concat((pd.read_csv(f) for f in fnames))
        self.experiment_name = np.unique(self.result_df['name'])[0]

    def plot_data(self,df,df_err=[],fname='plot',title='title',xlabel='xlabel',ylabel='ylabel',plot_min=0,plot_max=1,index_marker_shape=['o','v'],columns_marker_color=['m','g'],columns_title='Column_title',linestyle='--',scale_plot=True,x_scale_factor=1.0,x_scale_addition=0.0,plot_on_number_line=False):
        fig = plt.figure(dpi=100)
        ax = plt.gca()

        if plot_on_number_line:
            xi = list(df.index.values[1:])
        else:
            xi = list(range(len(df.index.values[1:])))
            
        plt.errorbar(xi,df.values[1:],yerr=0,capsize=5.0,label=None,marker='s',color='k')
        ax.axhline(y=1, xmin=0, xmax=1,linestyle='--',color='k')
        plt.xticks(xi, df.index.values[1:]*2)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if scale_plot:
            ax.set_ylim(plot_min, plot_max)

        x,y = np.array(xi),df.values[1:].squeeze()
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        print('slope = %f, t[4] = %f, p = %f' % (slope, intercept, p_value))
        plt.plot(x, y, 'o', label='original data')
        plt.plot(x, intercept + slope*x, 'r', label='fitted line')

        plt.tight_layout()

        plot_figure(fig,name=fname,caption='Results for '+self.experiment_name,experiment_name=self.experiment_name,dirname=self.dirname)
        plt.close(fig)

    def plot_data2(self,df,fname='plot',experiment_name='pelli',exp_name='LargeLetters',title='title',xlabel='xlabel',ylabel='ylabel',plot_min=0,plot_max=100,index_marker_shape=['o','v'],columns_marker_color=['m','g'],columns_title='Column_title',linestyle='--',scale_plot=True,show_target_as_dash=True,x_scale_factor=1.0,x_scale_addition=0.0,target_value=None,**params):
        if target_value is None:
            target_value = df.iloc[0].values[0]
        
        df = df[0:]

        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        if df.columns.values.shape[0] > 1:
            offset_color = columns_marker_color
        else:
            offset_color = ['k']
        marker_shape = index_marker_shape
        marker_color = offset_color[::-1]

        if 'plot_columns_labels' in params:
            columns_labels = params['plot_columns_labels']
        else:
            columns_labels = [str(x) for x in df.columns.values]

        fig = plt.figure(dpi=100)
        ax = plt.gca()

        for j,offset in enumerate(df.columns.values):
            c = offset_color[int(np.mod(j,len(offset_color)))]
            for i, row in enumerate(df.iterrows()):
                s = marker_shape[np.mod(i,len(marker_shape))]
                plt.plot((df.index[i]+x_scale_addition)*x_scale_factor,df.values[i,j],label=None,marker=s,linestyle=linestyle,markerfacecolor=c,markeredgecolor=c)
            plt.plot((df.index.values+x_scale_addition)*x_scale_factor,df.values[:,j],label=None,linestyle=linestyle,color=c)

        if show_target_as_dash:
            ax.axhline(y=target_value, xmin=0, xmax=1,linestyle=':',color='k')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        handles = []
        for j,offset in enumerate(df.columns.values.tolist()):
            c = offset_color[int(np.mod(j,len(offset_color)))]
            l = columns_labels[int(np.mod(j,len(columns_labels)))]
            patch = mpatches.Patch(color=c, label=l)
            handles.append(patch)
        if len(columns_labels) > 1:
            first_legend = plt.legend(title=columns_title,handles=handles,loc=4)
            plt.gca().add_artist(first_legend)
        handles = []
        for j,flank_dist in enumerate(np.round(np.array(df.index.values.tolist())*x_scale_factor,4)):
            s = marker_shape[int(np.mod(j,len(marker_shape)))]
            marker = mlines.Line2D([], [], color='k', marker=s, linestyle='None', markersize=10, label=str(flank_dist))
            handles.append(marker)

        plt.legend(title='Flank Dist.',handles=handles,framealpha=0.7,loc=8)

        if scale_plot:
            ax.set_ylim(plot_min, plot_max)

        plt.tight_layout()
        plot_figure(fig,name=fname,caption='Results for '+experiment_name+':'+exp_name,experiment_name=experiment_name,dirname=self.dirname)
        plt.close(fig)                                            
        
    def plot_data_as_barplot(self,df,fname='plot',title='title',xlabel='xlabel',ylabel='ylabel',target_value=1):
        df1 = df.copy()
        df1.index = np.array([chr(x) for x in np.arange(len(df.index.values))+97])   

        fig = plt.figure(dpi=100)
        ax = plt.gca()

        df1.plot.bar(ax=ax,color='k',rot=0,capsize=4,legend=False,yerr=5.0)
        ax.axhline(y=target_value, xmin=0, xmax=1,linestyle='--',color='k')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # if scale_plot:
        #     ax.set_ylim(plot_min, plot_max)

        plt.tight_layout()
        plot_figure(fig,name=fname,caption='Results for '+self.experiment_name,experiment_name=self.experiment_name,dirname=self.dirname)
        plt.close(fig)

    def plot_data_as_barplot2(self,df,fname='plot',experiment_name='pelli',exp_name='LargeLetters',title='title',xlabel='xlabel',ylabel='ylabel',columns_labels=None,plot_bar_min=0,plot_bar_max=40):
        base_contrast = df.values[0,0]
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter

        if columns_labels is None:
            columns_labels = [str(x) for x in df.columns.values]

        fig = plt.figure(dpi=100)
        ax = plt.gca()

        df = df[0:].T.groupby(['num_flank']).mean().T.mean()
        df = pd.DataFrame(np.matrix(np.hstack((base_contrast,df.values))).T,index=['0','1','5'],columns=['rrms'])
        df.plot.bar(ax=ax,style=['-','-','--'],color=['#959acd','#00b99c','#a2a5a8'],rot=0,capsize=4,ylim=[plot_bar_min,plot_bar_max],legend=False)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        plt.tight_layout()
        plot_figure(fig,name=fname,caption='Results for '+experiment_name+':'+exp_name,experiment_name=experiment_name,dirname=self.dirname)
        plt.close(fig)                                     

    def plot_decision_func(self,df,df_contrast=None,target_contrast=0.027,sigma=0.05,K=5.0,zoomed=False,fname='plot',experiment_name='pelli',exp_name='LargeLetters',title='title',xlabel='xlabel',ylabel='ylabel',index_marker_shape=['o','v'],columns_marker_color=['m','g'],columns_title='Column_title',scale_range=[0,1],upper_limit=0.85,lower_limit=0.0,x_scale_factor=1.0,show_target_as_dash=True,compute_relative_to_chance=False,chance=0.0,compute_error=False,plot_columns_labels=['0','1','2','3','4'],show_legend=False,decision_prob_label='response',est_max=0,transpose_df=True,use_target_contrast=False,legend_alpha=0.7,legend_loc=8,target_value=False):
        if compute_error:
            df = pd.DataFrame(df['contrast']).join(pd.DataFrame(1-df[decision_prob_label])) 

        if transpose_df:
            df_contrast = df['contrast'].unstack().T
            df = df[decision_prob_label].unstack().T
        else:
            df_contrast = df['contrast'].unstack()
            df = df[decision_prob_label].unstack()

        if not target_value:
            target_value = np.nanmax(df.values[:,0])
            if use_target_contrast:
                target_value = target_contrast

        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        if df.columns.values.shape[0] > 1:
            offset_color = columns_marker_color
        else:
            offset_color = ['k']
        marker_shape = index_marker_shape
        marker_color = offset_color[::-1]

        if plot_columns_labels is not None:
            columns_labels = plot_columns_labels
        else:
            columns_labels = [str(x) for x in df.columns.values]

        x = np.linspace(0,np.nanmax(df_contrast.values),100000)
        y = decision_func(x,target_contrast=target_contrast,decision_sigma=sigma,decision_K=K,upper_limit=upper_limit,lower_limit=lower_limit)

        if compute_relative_to_chance:
            y = ((y-chance) / (upper_limit - chance)) * upper_limit

        fig = plt.figure(dpi=100)
        ax = plt.gca()

        plt.plot(x,y,color='k',linestyle='--')

        for j,offset in enumerate(df.columns.values):
            c = offset_color[int(np.mod(j,len(offset_color)))]
            for i, row in enumerate(df.iterrows()):
                s = marker_shape[np.mod(i,len(marker_shape))]
                plt.plot(df_contrast.values[i,j],df.values[i,j],label=None,marker=s,linestyle='None',markerfacecolor=c,markeredgecolor=c)
        ax = plt.gca()

        if zoomed:
            ax.set_xlim(np.nanmin(df_contrast.values), np.nanmax(df_contrast.values))
            ax.set_ylim(np.nanmin(df.values), np.nanmax(df.values))
        else:
            if show_target_as_dash:
                ax.axhline(y=target_value, xmin=0, xmax=1,linestyle=':',color='k')
            ax.set_xlim(0.9*target_contrast, np.nanmax(df_contrast.values))  

        handles = []
        for j,offset in enumerate(df.columns.values.tolist()):
            c = offset_color[int(np.mod(j,len(offset_color)))]
            l = columns_labels[int(np.mod(j,len(columns_labels)))]
            patch = mpatches.Patch(color=c, label=l)
            handles.append(patch)
        if len(columns_labels) > 1:
            first_legend = plt.legend(title=columns_title,handles=handles,loc=4)
            plt.gca().add_artist(first_legend)
        handles = []
        for j,flank_dist in enumerate(np.round(np.array(df.index.values.tolist())*x_scale_factor,4)):
            s = marker_shape[int(np.mod(j,len(marker_shape)))]
            marker = mlines.Line2D([], [], color='k', marker=s, linestyle='None', markersize=10, label=str(flank_dist))
            handles.append(marker)

        if show_legend:
            plt.legend(title='Flank Dist.',handles=handles,framealpha=legend_alpha,loc=legend_loc)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        plt.tight_layout()
        plot_figure(fig,name=fname,caption='Results for '+experiment_name+':'+exp_name,experiment_name=experiment_name,dirname=self.dirname)
        plt.close(fig)
        
    def generate(self):
        def get_df(df,dvar='key_resp.corr'):
            df_result = pd.pivot_table(df,columns=['exp_num']+self.report_col_params,index=self.report_idx_params,values=[dvar],aggfunc=np.mean)
            df_result.columns = df_result.columns.droplevel() # remove extraneous outer level
            df_err = pd.pivot_table(df,columns=['exp_num']+self.report_col_params,index=self.report_idx_params,values=[dvar],aggfunc=sem)
            df_err.columns = df_err.columns.droplevel() # remove ext
            return df_result,df_err
            
        print('Generating report for: '+ self.experiment_name)

        # delete NA on human subject data
        dvar = 'key_resp.corr'
        result_df = self.result_df.dropna(axis=0,subset=['trials.thisN'])

        self.df_corr,self.df_err = get_df(result_df,dvar='key_resp.corr')
        df_decision = pd.pivot_table(result_df,columns=['exp_num'],index=self.report_col_params+self.report_idx_params,values=['contrast','decision_prob'],aggfunc=np.mean).swaplevel(0,1,axis=1)
        
        # eliminate the Nan's on each experiment
        self.df_decision = []
        for i in df_decision.columns.levels[0]:
            self.df_decision.append(df_decision[i].dropna())

        self._generate()

    def _generate(self):
        pass

    
def fit(df):
    if df.ndim == 3:
        mm = np.max(np.max(df[0]))
    else:
        mm = np.max(np.max(df))
    result = mm - df

    return result 

def sorted_ls(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    return list(sorted(os.listdir(path), key=mtime))[::-1]

import glob 
def main(reportclass=Report,filenames=None,only_most_recent=False,dirname=''):
    if len(filenames) == 0:
        filenames = glob.glob(os.path.join(dirname, 'data', 'model-*.csv'))

    r = reportclass(filenames=filenames,only_most_recent=only_most_recent,dirname=dirname+os.sep+'report')
    r.generate()
    
   
# mark functions to profile with @profile
# profile with: kernprof -l -v framework.py
if __name__ == "__main__":
    main(reportclass=Report,filenames=sys.argv[1:],only_most_recent=False,dirname=os.getcwd())



"""
=========================================================
Experiments Library
=========================================================

Contains functions that are used by all specific experiments.
"""

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Contrast.plotting import plot_figure

def decision_func(df,target_contrast=0.027,sigma=0.05,K=5.0,upper_limit=0.85,lower_limit=0.0):
    """
    Computes the result of applying the descision function to all the contrast values in the dataframe df.

    The decision function computes the difference of gaussians applied to the contrast values in df as follows:

    decision = :math:`e^{ - { ( \\frac{\\text{df} - \\text{target\_contrast}}{ 2 { \\sigma_t }^2  } ) }^2 } - e^{ - { ( \\frac{\\text{df} - \\text{target\_contrast}}{ 2 { K^2 \\sigma_t }^2  } ) }^2 }  + 1`

    :param df: the contast values for the experiment as a dataframe
    :param target_contrast: the contrast value for the target-alone condition
    :param sigma: the std. dev. of the target-alone contrast :math:`\\sigma_{t}`
    :param K: mulitplier for target+distractor sigma :math:`\\sigma_{t+d} = K\\sigma_t`
    :param upper_limit: upper_limit for the descision function which normally is 1.0
    :param lower_limit: lower_limit for the descision function which normally is 0.0
    :returns: result of decision function applied to dataframe df
    """
    J_f = np.exp(-(np.square(df-target_contrast)/(2.0*np.square(sigma))))
    J_b = np.exp(-(np.power((df-target_contrast)/(2.0*sigma*K),2)))
    result = (J_f-J_b+1)*(upper_limit-lower_limit)+lower_limit
    return result
                                                                        
def compute_response(df1,compute_error=False,est_max=-1.0,sigma_t=5.0,upper_limit=0.85,lower_limit=0.0):
    """returns df,mean_target,sigma_target,K

    :param df1: the dataframe used to compute responses, see note 1 below
    :param compute_error: if True, responses will be 1-response 
    :param est_max: method used to estimate the maximum background sigma
    :param sigma_t: divisor for sigma_t :math:`\\sigma_{t+d} = \\frac{est_max-\\mu_t}{\\sigma_t}`
    :returns: df,mean_target,sigma_target,K

    .. note:: 1: where df1 is of the form:

        .. code-block:: python

            # where df1 is of the form:
            offset                5         10        15        20
            flank_distance                                        
             -1.00           0.009318  0.010185  0.011334  0.012599
              0.05           0.010523  0.009813  0.010544  0.012038
              0.10           0.013262  0.012097  0.011458  0.011655
              0.15           0.013903  0.013449  0.013085  0.012946
              0.20           0.013882  0.013690  0.013561  0.013528
              0.30           0.013623  0.013646  0.014068  0.014614
              0.40           0.013587  0.013523  0.013959  0.014692
              0.60           0.013583  0.013481  0.013832  0.014533

    """
    
    values_target_per_offset = df1.iloc[0].values
    values_target_and_distractor_per_offset = df1[0:].values

    mean_target = np.mean(values_target_per_offset)
    mean_target_and_distractor = np.mean(values_target_and_distractor_per_offset)

    est_max = np.max(values_target_and_distractor_per_offset)
    sigma_target_and_distractor1 = (est_max-mean_target)/sigma_t
    vals = values_target_and_distractor_per_offset-np.mean(values_target_and_distractor_per_offset)+mean_target
    sigma_target_and_distractor2 = np.std(np.vstack((vals,-vals)))/3.0
    
    compute_K_with_est_max = True
    if compute_K_with_est_max:
        sigma_target_and_distractor = sigma_target_and_distractor1
    else:
        sigma_target_and_distractor = sigma_target_and_distractor2
       
    sigma_target_value = np.std(np.unique(values_target_per_offset))/3.0 # CURRENT 5/22/2019
    if sigma_target_value == 0.0:
        sigma_target_value = (0.01*mean_target)/3.0  # 3 std. from target is 1% increase in target value
    sigma_target = sigma_target_value

    K = sigma_target_and_distractor/sigma_target       
    K1 = sigma_target_and_distractor1/sigma_target
    K2 = sigma_target_and_distractor2/sigma_target

    df = decision_func(df1,target_contrast=mean_target,sigma=sigma_target,K=K,upper_limit=upper_limit,lower_limit=lower_limit)
    if compute_error:
        df = 1 - df
        
    df = df * 100.0 # for percent scale

    return df,mean_target,sigma_target,K

def plot_data_as_barplot(df,fname='plot',experiment_name='pelli',exp_name='LargeLetters',title='title',xlabel='xlabel',ylabel='ylabel',**params):
    base_contrast = df.values[0,0]
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    if 'columns_labels' in params:
        columns_labels = params['columns_labels']
    else:
        columns_labels = [str(x) for x in df.columns.values]

    fig = plt.figure(dpi=100)
    ax = plt.gca()
    
    df = df[0:].T.groupby(['num_flank']).mean().T.mean()
    df = pd.DataFrame(np.matrix(np.hstack((base_contrast,df.values))).T,index=['0','1','5'],columns=['rrms'])
    df.plot.bar(ax=ax,style=['-','-','--'],color=['#959acd','#00b99c','#a2a5a8'],rot=0,capsize=4,ylim=[params['plot_bar_min'],params['plot_bar_max']],legend=False)
   
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    plot_figure(fig,name=fname,caption='Results for '+experiment_name+':'+exp_name,experiment_name=experiment_name)
    plt.close(fig)                                     


def plot_data(df,fname='plot',experiment_name='pelli',exp_name='LargeLetters',title='title',xlabel='xlabel',ylabel='ylabel',plot_min=0,plot_max=100,index_marker_shape=['o','v'],columns_marker_color=['m','g'],columns_title='Column_title',linestyle='--',scale_plot=True,show_target_as_dash=True,x_scale_factor=1.0,x_scale_addition=0.0,**params):
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
    plot_figure(fig,name=fname,caption='Results for '+experiment_name+':'+exp_name,experiment_name=experiment_name)
    plt.close(fig)
                                      

def plot_decision_func(df,df_contrast=None,target_contrast=0.027,sigma=0.05,K=5.0,zoomed=False,fname='plot',experiment_name='pelli',exp_name='LargeLetters',title='title',xlabel='xlabel',ylabel='ylabel',index_marker_shape=['o','v'],columns_marker_color=['m','g'],columns_title='Column_title',scale_range=[0,1],upper_limit=0.85,lower_limit=0.0,x_scale_factor=1.0,show_target_as_dash=True,**params):
    if params['compute_error']:
        df=1-df
    target_value = df.iloc[0].values[0]
    df = df[0:]
    df_contrast = df_contrast[0:]
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

    x = np.linspace(0,np.max(df_contrast.values),100000)
    y = decision_func(x,target_contrast=target_contrast,sigma=sigma,K=K,upper_limit=upper_limit,lower_limit=lower_limit)
    
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
        ax.set_xlim(np.min(df_contrast.values), np.max(df_contrast.values))
        ax.set_ylim(np.min(df.values), np.max(df.values))
    else:
        if show_target_as_dash:
            ax.axhline(y=target_value, xmin=0, xmax=1,linestyle=':',color='k')
        ax.set_xlim(0.9*target_contrast, np.max(df_contrast.values))  

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
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
            
    plt.tight_layout()
    plot_figure(fig,name=fname,caption='Results for '+experiment_name+':'+exp_name,experiment_name=experiment_name)
    plt.close(fig)





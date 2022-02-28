import numpy as np
import pandas as pd
import scipy.stats as scs
import itertools
from collections import defaultdict
import textwrap

import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, OPTICS
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
pd.options.display.max_columns = 150

from umap import UMAP
from statannot import add_stat_annotation

import plotly
import plotly.graph_objs as go
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from mpl_toolkits import mplot3d
import matplotlib.cm as cm
from adjustText import adjust_text
import matplotlib.patheffects as PathEffects

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

###

def simple_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    
def run_by_group(orig_df,
                  **kwargs):
    g = orig_df.groupby(kwargs['groupby'])
    base_name = kwargs['save_name']
    
    for group, data in g:
        kwargs['title'] = group
        kwargs['save_name'] = base_name+'_'+group
        run_graph(data, **kwargs)
    return


def run_graph(df,
             **kwargs):
    fig, ax = plt.subplots(figsize=kwargs['figsize'])
        
    if 'violin' in kwargs['save_name']:
        ax = run_violin(df, ax, **kwargs)
    elif 'scatter' in kwargs['save_name']:
        if '5' in kwargs['save_name']:
            ax = run_scatter_5(df, ax, **kwargs)
        else:
            ax = run_scatter(df, ax, **kwargs)
        if 'comp_df' in kwargs:
            ax = run_loadings(df, ax, **kwargs)
    elif 'reg' in kwargs['save_name']:
        ax = run_scatter(df, ax, **kwargs)
    elif 'hist' in kwargs['save_name']:
        ax = run_hist(df, ax, **kwargs)
    elif 'bar' in kwargs['save_name']:
        ax = run_bar(df, ax, **kwargs)
    elif 'stacked' in kwargs['save_name']:
        ax = run_stacked_bar(df, ax, **kwargs)
    elif 'box' in kwargs['save_name']:
        ax = run_box(df, ax, **kwargs)
    elif 'sil' in kwargs['save_name']:
        ax = run_sil(df, ax, **kwargs)
    elif 'kde' in kwargs['save_name']:
        ax = run_kde(df, ax, **kwargs)
    elif 'line' in kwargs['save_name']:
        ax = run_line(df, ax, **kwargs)

    if 'log' in kwargs:
#         ax.set_xscale('symlog', linthreshx=1e-1)
#         ax.set_yscale('symlog', linthreshy=1e-1)
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    if 'xlims' in kwargs:
        if len(kwargs['xlims']) == 1:
            xlims = ax.get_xlim()
            kwargs['xlims'] = (kwargs['xlims'][0], xlims[1])
        ax.set_xlim(kwargs['xlims'])
        
    if 'ylims' in kwargs:
        if len(kwargs['ylims']) == 1:
            ylims = ax.get_ylim()
            kwargs['ylims'] = (kwargs['ylims'][0], ylims[1])
        ax.set_ylim(kwargs['ylims'])

    ax.set_xlabel(kwargs['x_label'], fontweight='bold', fontsize=11)
    ax.set_ylabel(kwargs['y_label'], fontweight='bold', fontsize=11)
    
    
#     if 'comp_df' in kwargs:
#         ax2 = ax.twiny()
#         ax2.set_xticks( ax.get_xticks() )
#         ax2.set_xbound(ax.get_xbound())
#         ax2.set_xticklabels([x/ax.get_xticks().max() for x in ax.get_xticks()])
#         ax2.set_xlabel('Loadings on PC'+str(kwargs['x']+1), fontweight='bold', fontsize=11)

#         ax3 = ax.twinx()
#         ax3.set_yticks(ax.get_yticks())
#         ax3.set_ybound(ax.get_ybound())
#         ax3.set_yticklabels([y/ax.get_yticks().max() for y in ax.get_yticks()])
#         ax3.set_ylabel('Loadings on PC'+str(kwargs['y']+1), fontweight='bold', fontsize=11)

    ax.set_title(kwargs['title'], fontweight='bold', fontsize=12)
    
    simple_axis(ax)
    plt.tight_layout()
    fig.savefig('viz/'+kwargs['save_name']+'.png')
    
    return


def sample_df(df, x):
    if x==1:
        return df
    elif x<1:
        return df.sample(frac=x, random_state=56)
    else:
        return df.sample(n=x, random_state=56)

    
def run_violin(data, ax, **kwargs):
    sub_df = sample_df(data, kwargs['sample_frac'])
    
    # get ns from full dataset
    if kwargs['x'] == 'region':
        df = pd.melt(data.loc[:, kwargs['cols']],
                    id_vars='region', value_vars='tot_thc').drop(columns=['variable'])
        sub_df = pd.melt(sub_data.loc[:, kwargs['cols']],
                    id_vars='region', value_vars='tot_thc').drop(columns=['variable'])
        order, n_dict = violin_order(df, group_by='region')
    else:
        if 'cols' in kwargs:
            df = pd.melt(data.loc[:, kwargs['cols']], var_name=kwargs['x'])
            sub_df = pd.melt(sub_df.loc[:, kwargs['cols']], var_name=kwargs['x'])
            order, n_dict = violin_order(df, group_by=kwargs['x'])
        else:
            df = data
            sub_df = sub_df
            order, n_dict = violin_order(df, group_by=kwargs['x'])

    # if pre-set order, use that
    if 'order' in kwargs:
        order = kwargs['order']

    # plot with sampled data
    if 'palette' in kwargs:
        sns.violinplot(x=kwargs['x'], y=kwargs['y'],
                       data=sub_df,
                       scale='width', order=order,
                       palette=kwargs['palette'], linewidth=0, ax=ax)
    else:
        sns.violinplot(x=kwargs['x'], y=kwargs['y'],
                       data=sub_df,
                       scale='width', order=order,
                       color='lightslategray', linewidth=0, ax=ax)
    
    PROPS = {
        'boxprops':{'facecolor':'black', 'edgecolor':'black', 'linewidth':3},
        'medianprops':{'color':'white', 'linewidth':2},
        'whiskerprops':{'color':'black', 'linewidth':2}
    }

    boxplot = sns.boxplot(x=kwargs['x'], y=kwargs['y'],
                          data=df, order=order,
                          showcaps=False, width=0.06,
                          fliersize=0.5, ax=ax, **PROPS)
    
    if kwargs['avg']:
        avg_avgs = df.groupby(kwargs['x'])[kwargs['y']].mean().mean()
        ax.axhline(avg_avgs, color='black', linestyle='--')
        
    if 'axhline' in kwargs:
        ax.axhline(kwargs['axhline'], color='black', linestyle='--')
        
    if 'sil-scores' in kwargs['save_name']:
        ax.axhline(0, color='black', linestyle='--')
    
    if kwargs['sig_comp']:
        box_pairs = list(itertools.combinations(order,r=2))
        
        test_results = add_stat_annotation(ax, data=df,
                                           x=kwargs['x'], y=kwargs['y'],
                                           order=order,
                                           box_pairs=box_pairs,
                                           text_annot_custom=[get_stats(df, pair, kwargs['x']) for pair in box_pairs],
                                           perform_stat_test=False, pvalues=[0, 0, 0],
                                           loc='outside', verbose=0)
                
#         ttest_df = pd.DataFrame(index=order, columns=['y_val','p_val','cohens_d'])
#         ttest_df[['y_val','p_val','cohens_d']] = ttest_df.apply(run_cohens, args=(df, ), axis=1, result_type='expand')
#         p_val_adj = 0.05/ttest_df.shape[0]
#         ttest_df['reject'] = ttest_df['p_val'] <= p_val_adj

#         bins = [0, 0.2, 0.5, 0.8, np.inf]
#         names = ['', '*', '**', '***']
#         ttest_df['star'] = pd.cut(np.abs(ttest_df['cohens_d']), bins, labels=names)
        
#         for i, region in enumerate(order):
#             if ttest_df.loc[region, 'reject']:
#                 y = ttest_df.loc[region, 'y_val']
                
#                 ax.text(i, y+2, ttest_df.loc[region, 'star'], ha='center', size=20)
    
    if 'v_xticklabels' in kwargs:
        xtick_labels = ax.get_xticklabels()
        labels = [textwrap.fill(x.get_text(),10) for x in xtick_labels]
        _ = ax.set_xticklabels(labels, rotation=90, ha='center')
    else:
        xtick_labels = ax.get_xticklabels()
        labels = [x.get_text()+'\nn='+str(n_dict[x.get_text()]['value']) for x in xtick_labels]
        _ = ax.set_xticklabels(labels)
    
    return ax


def violin_order(df, group_by='Cannab'):
    order = df.groupby(group_by).median().sort_values(by='value', ascending=False).index
    n_dict = df.groupby(group_by).count().T.to_dict(orient='dict')
    return order.values, n_dict


def run_scatter(df, ax, **kwargs):
    no_nan = df.dropna(subset=[kwargs['x'], kwargs['y']], how='any')
    sub_df = sample_df(no_nan, kwargs['sample_frac'])
    
    if 'size' in kwargs:
        s = kwargs['size']
    else:
        s = mpl.rcParams['lines.markersize']**2
        
    if 'edgecolor' in kwargs:
        ec = kwargs['edgecolor']
    else:
        ec = 'white'
    
    if 'hue' in kwargs:
        if 'sort_list' in kwargs:
            hue_order = kwargs['sort_list']
            sub_df = sub_df.sort_values(kwargs['hue'], key=make_sorter(kwargs['sort_list']))
        else:
            hue_order = sub_df[kwargs['hue']].value_counts().index
    
        sns.scatterplot(x=kwargs['x'], y=kwargs['y'],
                        hue=kwargs['hue'],
                        data=sub_df,
                        s=s,
                        edgecolor=ec,
                        alpha=0.5,
                        hue_order=hue_order,
                        palette=kwargs['palette'],
                        ax=ax)
         # include full ns
        handles, labels = ax.get_legend_handles_labels()

        if 'n_display' in kwargs:
            labels_n = [(cat, df.loc[df[kwargs['hue']]==cat].shape[0]) for cat in hue_order]
            labels = [cat+'\nn='+str(n) for cat, n in labels_n]

            ax.legend(handles=handles[:kwargs['n_display']], labels=labels[:kwargs['n_display']], title=kwargs['hue'].title(), handlelength=4)
        else:
            labels_n = [(cat, df.loc[df[kwargs['hue']]==cat].shape[0]) for cat in hue_order]
            labels = [cat+'\nn='+str(n) for cat, n in labels_n]

            ax.legend(handles=handles, labels=labels, title=kwargs['hue'].title(), handlelength=4)
    else:
        sns.regplot(x=kwargs['x'], y=kwargs['y'],
                    data=sub_df,
                    scatter_kws={'alpha':0.1, 'color':'lightslategray', 'rasterized':True},
                    line_kws={'color':'orange'},
                    ax=ax)
        
        r, p = scs.spearmanr(no_nan[kwargs['x']], no_nan[kwargs['y']])
        labels = ['rho = {:.2f}'.format(r)]
    
        if p < 1e-300:
            labels.append('p < 1e-300')
        else:
            labels.append('p = {:.1e}'.format(p))
        ax.legend(labels=['\n'.join(labels)])
        
    if 'prod_strains' in kwargs:
        s_colors = ['black', 'gray', 'white']
        s_markers = ['^', 'D', 'o']
        n_strains = len(kwargs['prod_strains'])
        for strain, color, marker in zip(kwargs['prod_strains'], s_colors[:n_strains], s_markers[:n_strains]):
            sns.scatterplot(x=kwargs['x'], y=kwargs['y'],
                data=sub_df.loc[sub_df['strain_slug']==strain],
                s=s+35,
                edgecolor='black',
                linewidth=1.5,
                color=color,
                label=strain,
                marker=marker,
                ax=ax)
    return ax


def get_log(df, cannab_1='tot_thc', cannab_2='tot_cbd'):    
    # get THC_CBD ratio for batches without 0 tot_thc (avoid dividing by 0)
    df['ratio'] = 0
    df.loc[df[cannab_2] != 0, 'ratio'] = (df.loc[df[cannab_2] != 0, cannab_1]) / (df.loc[df[cannab_2] != 0, cannab_2])
    
    # get log_THC_CBD vals 
    df['log_ratio'] = 0
    df.loc[df['ratio'] != 0, 'log_ratio'] = np.log10(df.loc[df['ratio'] != 0, 'ratio'])
    
    # set the 0 tot_cbd batches to an extraneous high bin
    df.loc[df[cannab_2] == 0, 'log_ratio'] = 4
    df.loc[df[cannab_1] == 0, 'log_ratio'] = -2
    
    log_ratio = df['log_ratio']
    
    return log_ratio


def run_hist(df, ax, **kwargs):
    sub_df = sample_df(df, kwargs['sample_frac'])
    
    # some cut-offs
    ct_thresh_high = 5
    ct_thresh_low = 0.25
    max_log = 4
    min_log = -2.0
    
    # get log data
    log_cannab = get_log(sub_df, cannab_1='tot_'+kwargs['x'], cannab_2='tot_'+kwargs['y'])
    
    # get histogram
    hist, bins = np.histogram(log_cannab, bins=np.arange(min_log-0.1, max_log+0.1, 0.05))

    # get colors
    colors = []
    for low, high in zip(bins,bins[1:]):
        avg = np.mean([low, high])
        if avg >= np.log10(ct_thresh_high):
            colors.append('darkblue')
        elif avg <= np.log10(ct_thresh_low):
            colors.append('black')
        else:
            colors.append('steelblue')
    
    # plot histogram, thresholds
    ax.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), 
       width=(bins[1]-bins[0]), color=colors)        
    ax.plot([np.log10(ct_thresh_high), np.log10(ct_thresh_high)], [0, kwargs['ylims'][1]-0.02], 
            linestyle='--', color='k', linewidth=1)
    ax.plot([np.log10(ct_thresh_low), np.log10(ct_thresh_low)], [0, kwargs['ylims'][1]-0.02], 
            linestyle='--', color='k', linewidth=1)

    ax.set_xticklabels(['',float("-inf"), -1, 0, 1, 2, 3, float("inf")]) 

    # draw legend
    chemotypes = ['THC-Dom', 'Bal THC/CBD', 'CBD-Dom']
                  
    ct_1 = mpatches.Patch(color='darkblue', label='THC-Dom')
    ct_2 = mpatches.Patch(color='steelblue', label='Bal THC/CBD')
    ct_3 = mpatches.Patch(color='black', label='CBD-Dom')
                  
    ct_handles, ct_labels = ax.get_legend_handles_labels()
    
    ct_labels_n = [(x, df.loc[df['chemotype']==x].shape[0]) for x in chemotypes]
    ct_labels = [x+'\nn='+str(n) for x, n in ct_labels_n]
    
    ax.legend(handles=[ct_1,ct_2,ct_3], labels=ct_labels,title='Chemotype',handlelength=4)
    
    return ax


def normalize(df, cols):
    df.loc[:, cols] = (df.loc[:, cols]
                       .div(df.loc[:, cols].sum(axis=1), axis=0)
                       .multiply(100))
    return df


def max_min(arr):
    return arr/(arr.max()-arr.min())


def make_sorter(sort_list):
    """
    Create a dict from the list to map to 0..len(l)
    Returns a mapper to map a series to this custom sort order
    """
    sort_order = {k:v for k,v in zip(sort_list, range(len(sort_list)))}
    return lambda s: s.map(lambda x: sort_order[x])


def run_bar(df, ax, **kwargs):
    if 'hue' in kwargs:
        sns.barplot(x=kwargs['x'], y=kwargs['y'],
                    hue=kwargs['hue'],
                    data=df,
                    palette=kwargs['palette'],
                    order=kwargs['order'])
    elif 'palette' in kwargs:
        sns.barplot(x=kwargs['x'], y=kwargs['y'],
                    data=df,
                    palette=kwargs['palette'],
                    order=kwargs['order'])
    else:
        sns.barplot(x=kwargs['x'], y=kwargs['y'],
                   color='lightslategray')

    return ax


def run_box(df, ax, **kwargs):
    if 'palette' in kwargs:
        sns.boxplot(x=kwargs['x'], y=kwargs['y'],
                    data=df,
                    palette=kwargs['palette'],
                    order=kwargs['order'])
    else:
        sns.boxplot(x=kwargs['x'], y=kwargs['y'],
               color='lightslategray')
    
    return ax



def run_pca(df, cols, norm=True, n_components=2, max_min_arr=False):
    df[cols] = df[cols].fillna(0)
    
    # get rid of rows that are all 0 for specified columns
    zero_bool = (df[cols]==0).sum(axis=1)==len(cols)
    df = df[~zero_bool].copy()
    
    if norm:
        X = normalize(df, cols).copy()
    else:
        X = df.copy()
        
    model = PCA(n_components=n_components)
    model.fit(X.loc[:, cols])
    arr = model.fit_transform(X.loc[:, cols])
    
    if max_min_arr:
        arr = np.apply_along_axis(max_min, arr=arr, axis=0)
    
    # add first three component scores to df
    X[0] = arr[:,0]
    X[1] = arr[:,1]
    X[2] = arr[:,2]
    
    return X, arr, model


def run_loadings(df, ax, **kwargs):
    comp_df = kwargs['comp_df']
    comp_df['combo_score'] = np.abs(comp_df[[kwargs['x'],kwargs['y']]]).sum(axis=1)
    comp_df = comp_df.sort_values(by='combo_score', ascending=False).iloc[:kwargs['n_display']]
    
    max_x = df[kwargs['x']].max()
    max_y = df[kwargs['y']].max()
    
    texts = []
    for x in comp_df.iterrows():
        texts.append(ax.text(x[1][kwargs['x']]*max_x,
                             x[1][kwargs['y']]*max_y,
                             x[0],
                             fontweight='bold',
                             bbox=dict(facecolor='white', edgecolor='blue', pad=2, alpha=0.75)))
        ax.arrow(0, 0,
                 x[1][kwargs['x']]*max_x,
                 x[1][kwargs['y']]*max_y,
                 color='black', alpha=1,
                lw=2, head_width=1)
    adjust_text(texts)
    
    return ax


def run_sil(df, ax, **kwargs):
    sub_df = sample_df(df, kwargs['sample_frac'])
    
    labels = df[kwargs['hue']]
    sub_labels = sub_df[kwargs['hue']]
    label_list = labels.value_counts().index
    
    silhouette_avg = silhouette_score(df[kwargs['cols']], labels)
    sample_sil_val = silhouette_samples(sub_df[kwargs['cols']], sub_labels)
    
    y_lower=0
    for i, label in enumerate(label_list[:kwargs['n_display']]):
        ith_cluster_sil_val = sample_sil_val[sub_labels==label]
        ith_cluster_sil_val.sort()

        size_cluster_i = ith_cluster_sil_val.shape[0]
        y_upper = y_lower+size_cluster_i

        color = kwargs['palette'][label]
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                           0, ith_cluster_sil_val,
                           facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower+0.5*size_cluster_i, label)
        y_lower = y_upper+1
    
    ax.axvline(silhouette_avg, color='lightslategray', linestyle='--')
    
    ax.legend(labels=['Avg Silhouette Score {:.2f}'.format(silhouette_avg)])

    ax.set_ylim(0, y_upper+10)

    return ax


def score2loading(x):
    return x / x.max()


def loading2score(x):
    return x * x.max()


def get_ct(df):
    # determine THC/CBD ratio
    df['chemotype_ratio'] = df['tot_thc'].div(df['tot_cbd'], fill_value=0)
    df.loc[(df['tot_thc']==0)&(df['tot_cbd']!=0), 'chemotype_ratio'] = -np.inf
    df.loc[(df['tot_thc']!=0)&(df['tot_cbd']==0), 'chemotype_ratio'] = np.inf

    # bin chemotypes by ratio
    df['chemotype'] = pd.cut(df['chemotype_ratio'],
                                [-np.inf, 0.2, 5, np.inf],
                                 labels=['CBD-Dom','Bal THC/CBD', 'THC-Dom'],
                            include_lowest=True)
    return df


def run_stacked_bar(df, ax, **kwargs):
    if 'order' in kwargs:
        df[kwargs['order']].plot(kind='bar', stacked=True, color=kwargs['palette'], ax=ax)
    else:
        df.plot(kind='bar', stacked=True, color=kwargs['palette'], ax=ax)
    
    # .patches is everything inside of the chart
    for rect in ax.patches:
        # Find where everything is located
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()

        # The height of the bar is the data value and can be used as the label
        label_text = f'{height:.1f}%'  # f'{height:.2f}' to format decimal values

        # ax.text(x, y, text)
        label_x = x + width / 2
        label_y = y + height / 2

        # plot only when height is greater than specified value
        if height > 5:
            txt = ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=10, fontweight='bold', color='black')
            txt.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='w')])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)    

    
    return ax


# def run_polar_plot(df, ax, **kwargs):
# #     print(df.loc[:,kwargs['cols']])
#     mean_cols = list(df.loc[:,kwargs['cols']].mean(axis=0))
#     mean_cols2 = mean_cols + mean_cols[:1]
    
#     angles = [n / len(mean_cols) * 2 * np.pi for n in range(len(mean_cols))]
#     angles = angles + angles[:1]
    
#     # get color
#     order = np.argsort(mean_cols)[::-1]
#     top_val = np.array(kwargs['cols'])[order][0]
    
#     if 'colors' in kwargs:
#         colors = kwargs['colors']
#     else:
#         colors = kwargs['palette'][top_val]
    
# #     # error bars
# #     err_cols = list(df.loc[:,kwargs['cols']].std(axis=0))
# #     err_cols2 = err_cols + err_cols[:1]
# #     ax.errorbar(angles, mean_cols2, yerr=err_cols2, capsize=0, color=colors, linestyle='solid', ecolor='lightslategray')
    
#     # plot
#     if kwargs['avg']:
#         ax.plot(angles, mean_cols2, color=colors, lw=1, linestyle='solid')
#         ax.fill(angles, mean_cols2, colors, alpha=0.1)

#         # y limits
#         ax.set_ylim(0, np.max(mean_cols2))
#     else:
#         for row_idx, row in df[kwargs['cols']].iterrows():
#             row_list = list(row)
#             row_list2 = row_list + row_list[:1]
#             if type(colors)==str:
#                 ax.plot(angles, row_list2, color=colors, lw=0.5)
#             else:
#                 ax.plot(angles, row_list2, color=colors[row_idx], lw=0.5)
#             ax.set_ylim(0, np.max(df[kwargs['cols']].max()))
    

#     # tick labels
#     tick_labs = kwargs['cols']
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(tick_labs, color='black', size=10)
#     ax.set_yticks([])
    
#     return ax


def run_polar_plot(df, ax, **kwargs):
#     print(df.loc[:,kwargs['cols']])
    mean_cols = list(df.loc[:,kwargs['cols']].mean(axis=0))
    mean_cols2 = mean_cols + mean_cols[:1]
    
    angles = [n / len(mean_cols) * 2 * np.pi for n in range(len(mean_cols))]
    angles = angles + angles[:1]
    
    # plot samples
    if 'sub_n' in kwargs:
        sub_data = df.sort_values('n_samps', ascending=False)[:kwargs['sub_n']]
    else:
        sub_data = df
    
    for row_idx, row in sub_data[kwargs['cols']].iterrows():
        row_list = list(row)
        row_list2 = row_list + row_list[:1]
        if type(kwargs['colors'])==str:
            ax.plot(angles, row_list2, color=kwargs['colors'], lw=0.5, alpha=0.5)
        else:
            ax.plot(angles, row_list2, color=kwargs['colors'][row_idx], lw=0.5)
    
    if kwargs['avg']:
        # get for average color
        order = np.argsort(mean_cols)[::-1]
        top_val = np.array(kwargs['cols'])[order][0]

        avg_color = kwargs['palette'][top_val]
        
        ax.plot(angles, mean_cols2, color=avg_color, lw=1, linestyle='solid', zorder=11)
        ax.fill(angles, mean_cols2, avg_color, alpha=0.5, zorder=10)
        ax.set_ylim(0, np.max(mean_cols2))
    else:
        ax.set_ylim(0, np.max(sub_data[kwargs['cols']].max()))
    
    
    # tick labels
    tick_labs = kwargs['cols']
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tick_labs, color='black', size=10)
    ax.set_yticks([])
    
    return ax


def run_pairwise(df,
                 cann_cols, terp_cols):
    df_sim = pd.DataFrame(columns=['cann','terp','all'])
    for idx, cols in enumerate([cann_cols, terp_cols, cann_cols+terp_cols]):
        if idx==2:
            df[cols] = MinMaxScaler().fit_transform(df[cols].fillna(0))
        sim_scores = cosine_similarity(df[cols].fillna(0))
        sim_scores[sim_scores > 0.9999999999999] = np.nan
        df_sim.iloc[:, idx] = np.nanmean(sim_scores, axis=0)
    return df_sim


def get_scaled_dfs(df, cann, terps):
    X_cann = df[cann].fillna(0)
    X_cann_standard = MinMaxScaler().fit_transform(X_cann)

    X_terps = df[terps].fillna(0)
    X_terps_standard = MinMaxScaler().fit_transform(X_terps)

    X_all = df[cann+terps].fillna(0)
    X_all_standard = MinMaxScaler().fit_transform(X_all)
    
    return X_cann, X_cann_standard, X_terps, X_terps_standard, X_all, X_all_standard


def avg_sd(a,b):
    num = np.var(a)+np.var(b)
    return np.sqrt(num/2)


def get_stats(df, pair, col, return_vals=False):
    x = df.loc[df[col]==pair[0], 'value']
    y = df.loc[df[col]==pair[1], 'value']

    ttest = scs.ttest_ind(x, y, equal_var=False)
    d_prime = (x.mean()-y.mean())/avg_sd(x,y)
    
    if return_vals:
        return np.array([ttest[1],d_prime])
    else:
        labels = []
        if ttest[1] < 1e-300:
            labels.append('p < 1e-300')
        else:
            labels.append('p = {:.1e}'.format(ttest[1]))
        labels.append("d'="+str(np.round(np.abs(d_prime),2)))
        return ', '.join(labels)
    

    
def get_prod_df(df, n_samp_min, n_prod_min,
               common_cannabs,
               common_terps):
    n_samp_df = df.groupby(['anon_producer','strain_slug'])['u_id'].count()
    df = df.merge(n_samp_df.rename('n_samp'), left_on=['anon_producer','strain_slug'], right_index=True)
    
    # create producer df
    prod_df = df.loc[(df['n_samp']>=n_samp_min)].groupby(['anon_producer','strain_slug'])[common_cannabs+common_terps].mean()
    prod_df = get_ct(prod_df)
    prod_df = prod_df.reset_index(drop=False)

    # get n_prod counts
    n_prod_df = prod_df.groupby('strain_slug')['anon_producer'].count()
    prod_df = prod_df.merge(n_prod_df.rename('n_prod'), left_on='strain_slug', right_index=True)
    prod_df = prod_df.merge(n_samp_df.rename('n_samps'), left_on=['anon_producer','strain_slug'], right_index=True)
    
    # subset to n_prod_min and thc-dom
    fin_prod_df = prod_df.loc[(prod_df['n_prod']>=n_prod_min)].sort_values(['n_prod','strain_slug','anon_producer'], ascending=[False, False, True]).copy()
    fin_prod_df['strain_slug'] = fin_prod_df['strain_slug'].astype(str)
    
    return fin_prod_df


def get_pal_dict(df, common_terps, terp_dict):
    pal_dict = {}
    for label in set(df['kmeans_label']):
        terp_order = df.loc[df['kmeans_label']==label, common_terps].mean().sort_values(ascending=False)
        pal_dict[label] = terp_dict[terp_order[:1].index[0]]
    return pal_dict


def get_kmeans(df, common_terps,
              k=3):
    df_norm, arr, model = run_pca(df, common_terps, norm=True, n_components=3)

    # set up kmeans
    clust = KMeans(3, random_state=56)

    # get cluster labels
    df_norm['kmeans_label'] = clust.fit_predict(df_norm[common_terps])
    clust_dict = {x:y for x,y in zip(df_norm['kmeans_label'].value_counts().index, ['A','B','C'])}
    df_norm['kmeans_label'] = df_norm['kmeans_label'].replace(clust_dict)
    
    return df_norm


def get_umap(df, common_terps,
            n_neighbors=6,
            random_state=56):
    umap_ = UMAP(n_components=2, n_neighbors=n_neighbors,
                 random_state=random_state)

    X_terps_umap = umap_.fit_transform(df[common_terps])

    df['umap_0'] = X_terps_umap[:,0]
    df['umap_1'] = X_terps_umap[:,1]
    
    return df


def get_round(arr, sig_fig=1):
    return np.round(arr*100,sig_fig)


def get_cos_sim(df, add_nan=False):
    sim_scores = cosine_similarity(df)
    if add_nan:
        sim_scores[sim_scores > 0.9999999999999] = np.nan
    else:
        sim_scores[sim_scores > 0.9999999999999] = 1
    return sim_scores


def group_cos_sim(df, group_level=False):
    if df.shape[0]==1:
        # if only one product, do not return cos sim
        return np.nan
    else:
        sim_scores = get_cos_sim(df, add_nan=True)
        if group_level:
            return np.mean(np.nanmean(sim_scores, axis=0))
        else:
            return list(np.nanmean(sim_scores, axis=0))
        
        
def format_df(df):
    return df.explode().to_frame().rename(columns={0:'bw_prod_sim'}).reset_index(drop=False)


def weighted_avg(avgs, weights):
    return np.average(avgs, weights=weights)


def run_all_cos_sims(df, cols,
                    groupby='strain_slug'):
    groups = df.groupby(groupby)[cols]
    bw_prod_df = format_df(groups.apply(lambda x: group_cos_sim(x)))
    
    avgs = groups.apply(lambda x: group_cos_sim(x, group_level=True))
    weights = groups.size()[groups.size()>1]
    
    return bw_prod_df, avgs, weights



def run_kde(df, ax, **kwargs):
    no_nan = df.dropna(subset=[kwargs['x'], kwargs['y']], how='any')
    sub_df = sample_df(no_nan, kwargs['sample_frac'])
    
    _ = sns.kdeplot(x=kwargs['x'],
                    y=kwargs['y'],
               data=sub_df,
               fill=True,
               cmap='RdBu_r',
               cbar=True,
               vmin=0,
               levels=75,
               ax=ax)
    
    return ax


def run_line(df, ax, **kwargs):
    _ = ax.plot(kwargs['x'], kwargs['y'])
    return ax



def run_scatter_5(df, ax, **kwargs):
    no_nan = df.dropna(subset=[kwargs['x'], kwargs['y']], how='any')
    sub_df = sample_df(no_nan, kwargs['sample_frac'])
    
    if 'size' in kwargs:
        s = kwargs['size']
    else:
        s = mpl.rcParams['lines.markersize']**2
        
    if 'edgecolor' in kwargs:
        ec = kwargs['edgecolor']
    else:
        ec = 'white'

    hue_order = ['THC-Dom', 'Bal THC/CBD', 'CBD-Dom']
    s_colors = ['darkblue', 'steelblue', 'black']
    s_markers = ['D', '^', 'o']
    
    for ct, color, marker in zip(hue_order, s_colors, s_markers):
        if ct=='THC-Dom':
            sns.scatterplot(x=kwargs['x'], y=kwargs['y'],
                            data=sub_df.loc[df['chemotype']==ct], alpha=.5,
                            color=color,
                            marker=marker,
                            s=25,
                            edgecolor='white',
                            linewidth=0.5,
                            label=ct,
                            ax=ax)
        else:
            sns.scatterplot(x=kwargs['x'], y=kwargs['y'],
                            data=sub_df.loc[df['chemotype']==ct], alpha=1,
                            color=color,
                            marker=marker,
                            s=25,
                            edgecolor='white',
                            linewidth=0.5,
                            label=ct,
                            ax=ax)  

     # include full ns
    handles, labels = ax.get_legend_handles_labels()

    if 'n_display' in kwargs:
        labels_n = [(cat, df.loc[df[kwargs['hue']]==cat].shape[0]) for cat in hue_order]
        labels = [cat+'\nn='+str(n) for cat, n in labels_n]

        ax.legend(handles=handles[:kwargs['n_display']], labels=labels[:kwargs['n_display']], title=kwargs['hue'].title(), handlelength=4)
    else:
        labels_n = [(cat, df.loc[df[kwargs['hue']]==cat].shape[0]) for cat in hue_order]
        labels = [cat+'\nn='+str(n) for cat, n in labels_n]

        ax.legend(handles=handles, labels=labels, title=kwargs['hue'].title(), handlelength=4)
    
    return ax


def pearson_sig_string(s1,s2):
    _r,_p = scs.pearsonr(s1.fillna(0),s2.fillna(0))
    if _p < 0.001:
        return 'r={0:.2f}***'.format(_r)
    elif _p < 0.01:
        return 'r={0:.2f}**'.format(_r)
    elif _p < 0.05:
        return 'r={0:.2f}*'.format(_r)
    else:
        return 'r={0:.2f}'.format(_r)
    
def spearman_sig_string(s1,s2):
    _r,_p = scs.spearmanr(s1.fillna(0),s2.fillna(0))
    if _p < 0.0001/9:
        return 'r={0:.2f}***'.format(_r)
    elif _p < 0.001/9:
        return 'r={0:.2f}**'.format(_r)
    elif _p < 0.01/9:
        return 'r={0:.2f}*'.format(_r)
    else:
        return 'r={0:.2f}'.format(_r)

def run_fig2(df):
    f,axs = plt.subplots(3,3,figsize=(12,12))

    _lowess = False
    _robust = False
    _ci = 0 
    _size = 3

    # THC-Dom
    _df = df.loc[(df['chemotype']=='THC-Dom'), ['tot_thc','tot_cbd','tot_cbg','chemotype']]
    _df_samp = _df.sample(n=20000)
    _df_ol = _df_samp.loc[_df_samp['ol_fig2']==False]
    
    ## thc vs cbd
    sns.scatterplot(x='tot_thc', y='tot_cbd', data=_df_samp, marker='.', color='red', scatter_kws={'s':_size}, ax=axs[0,0])
    sns.regplot(x='tot_thc',y='tot_cbd',data=_df_ol,marker='.',color='darkblue',scatter_kws={'s':_size},line_kws={'color':'k','linewidth':2},lowess=_lowess,robust=_robust,ci=_ci,ax=axs[0,0])
    
    ## cbg vs thc
    sns.regplot(x='tot_cbg',y='tot_thc',data=_df_samp,marker='.',color='darkblue',scatter_kws={'s':_size},line_kws={'color':'k','linewidth':2},lowess=_lowess,robust=_robust,ci=_ci,ax=axs[1,0])
    ## cbd vs cbg
    sns.scatterplot(x='tot_cbg', y='tot_thc', data=_df_samp, marker='.', color='red', scatter_kws={'s':_size}, ax=axs[0,0])
    sns.regplot(x='tot_cbd',y='tot_cbg',data=_df_samp,marker='.',color='darkblue',scatter_kws={'s':_size},line_kws={'color':'k','linewidth':2},lowess=_lowess,robust=_robust,ci=_ci,ax=axs[2,0])

    axs[0,0].text(0.05,0.9,spearman_sig_string(_df['tot_thc'],_df['tot_cbd']),ha='left',va='center',transform=axs[0,0].transAxes,fontsize=15)
    axs[1,0].text(0.6,0.05,spearman_sig_string(_df['tot_cbg'],_df['tot_thc']),ha='left',va='center',transform=axs[1,0].transAxes,fontsize=15)
    axs[2,0].text(0.05,0.9,spearman_sig_string(_df['tot_cbd'],_df['tot_cbg']),ha='left',va='center',transform=axs[2,0].transAxes,fontsize=15)

    # Balanced
    _df = lab_df[['tot_thc','tot_cbd','tot_cbg','chemotype']].query("chemotype=='Bal THC/CBD'")# & tot_cbg < 5 & tot_cbd < 20")
    _df_samp = _df.sample(frac=1)
    sns.regplot(x='tot_thc',y='tot_cbd',data=_df,marker='.',color='steelblue',scatter_kws={'s':_size},line_kws={'color':'k','linewidth':2},lowess=_lowess,robust=_robust,ci=_ci,ax=axs[0,1])
    sns.regplot(x='tot_cbg',y='tot_thc',data=_df,marker='.',color='steelblue',scatter_kws={'s':_size},line_kws={'color':'k','linewidth':2},lowess=_lowess,robust=_robust,ci=_ci,ax=axs[1,1])
    sns.regplot(x='tot_cbd',y='tot_cbg',data=_df,marker='.',color='steelblue',scatter_kws={'s':_size},line_kws={'color':'k','linewidth':2},lowess=_lowess,robust=_robust,ci=_ci,ax=axs[2,1])

    axs[0,1].text(0.05,0.9,spearman_sig_string(_df['tot_thc'],_df['tot_cbd']),ha='left',va='center',transform=axs[0,1].transAxes,fontsize=15)
    axs[1,1].text(0.05,0.9,spearman_sig_string(_df['tot_cbg'],_df['tot_thc']),ha='left',va='center',transform=axs[1,1].transAxes,fontsize=15)
    axs[2,1].text(0.05,0.9,spearman_sig_string(_df['tot_cbd'],_df['tot_cbg']),ha='left',va='center',transform=axs[2,1].transAxes,fontsize=15)

    # CBD-Dom
    _df = lab_df[['tot_thc','tot_cbd','tot_cbg','chemotype']].query("chemotype=='CBD-Dom'")# & tot_cbg < 5 & tot_cbd < 20")
    _df_samp = _df.sample(frac=1)
    sns.regplot(x='tot_thc',y='tot_cbd',data=_df,marker='.',color='black',scatter_kws={'s':_size},line_kws={'color':'k','linewidth':2},lowess=_lowess,robust=_robust,ci=_ci,ax=axs[0,2])
    sns.regplot(x='tot_cbg',y='tot_thc',data=_df,marker='.',color='black',scatter_kws={'s':_size},line_kws={'color':'k','linewidth':2},lowess=_lowess,robust=_robust,ci=_ci,ax=axs[1,2])
    sns.regplot(x='tot_cbd',y='tot_cbg',data=_df,marker='.',color='black',scatter_kws={'s':_size},line_kws={'color':'k','linewidth':2},lowess=_lowess,robust=_robust,ci=_ci,ax=axs[2,2])

    axs[0,2].text(0.05,0.9,spearman_sig_string(_df['tot_thc'],_df['tot_cbd']),ha='left',va='center',transform=axs[0,2].transAxes,fontsize=15)
    axs[1,2].text(0.05,0.9,spearman_sig_string(_df['tot_cbg'],_df['tot_thc']),ha='left',va='center',transform=axs[1,2].transAxes,fontsize=15)
    axs[2,2].text(0.05,0.9,spearman_sig_string(_df['tot_cbd'],_df['tot_cbg']),ha='left',va='center',transform=axs[2,2].transAxes,fontsize=15)

    # Label axes
    axs[0,0].set_ylabel('CBD',fontsize=15)
    axs[1,0].set_ylabel('THC',fontsize=15)
    axs[2,0].set_ylabel('CBG',fontsize=15)

    axs[2,0].set_xlabel('CBD',fontsize=15)
    axs[2,1].set_xlabel('THC',fontsize=15)
    axs[2,2].set_xlabel('CBG',fontsize=15)

    for ax in axs[0,:]:
        ax.set_xlabel('THC',fontsize=15)

    for ax in axs[1,:]:
        ax.set_xlabel('CBG',fontsize=15)

    for ax in axs[2,:]:
        ax.set_xlabel('CBD',fontsize=15)

    for ax in axs[:,1:].flatten():
        ax.set_ylabel(None)

    axs[0,0].set_title('THC-Dominant',color='darkblue',fontsize=15,fontweight='bold')
    axs[0,1].set_title('Balanced',color='steelblue',fontsize=15,fontweight='bold')
    axs[0,2].set_title('CBD-Dominant',color='black',fontsize=15,fontweight='bold')

    f.tight_layout()
    f.savefig('cannabinoid_correlation.png',dpi=300)
    # f.savefig('cannabinoid_correlation.eps',dpi=300)
    # f.savefig('cannabinoid_correlation.pdf',dpi=300)
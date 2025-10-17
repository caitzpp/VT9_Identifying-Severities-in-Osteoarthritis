import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import math
from utils.load_utils import load_image

from scipy.stats import entropy, kruskal, combine_pvalues, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_fscore_support, f1_score, cohen_kappa_score, average_precision_score, normalized_mutual_info_score


def drop_unnamedcolumn(df):
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    return df

def get_nan_values(df):
    nan_columns = df.columns[df.isna().any()].tolist()
    nan_summary = df.isna().sum()

    print("Columns with NaN values: ", nan_columns)
    print()
    print("NaN values per column:")
    print(nan_summary[nan_summary > 0].sort_values(ascending=False))
    return nan_columns

def check_duplicate_rows(df, column = 'record_id'):
    dupl_recordid = pd.DataFrame(df[column].value_counts().reset_index())
    dupl_recordid_l = list(dupl_recordid[dupl_recordid['count']>1][column])

    #TODO: add logic to delete duplicates, where the entire row is the same and 
    # call out the duplicates that are not the same

    if len(dupl_recordid_l)>0:
        dupl_rows = df[df[column].isin(dupl_recordid_l)].sort_values(by=column)

        print(f"Duplicate rows based on {column}:")
        display(dupl_rows)
        return dupl_rows
    else:
        print(f"No duplicate rows based on {column}.")
        return None
    
def investigate_data(df, id_col = 'record_id', save_path = None, save_name = None):
    df_descr = df.describe(include='all')

    if save_path:
        if save_name is None:
            print("No save name provided!")
            return
        save_path = os.path.join(save_path, f"{save_name}_description.csv")
        df_descr.to_csv(save_path, index=True, mode='x')
        print(f"Data description saved to {save_path}")
        print()
    
    # max_count = df_descr.loc['count'].max()

    # for col in df_descr.columns:
    #     if df_descr[col]['count'] < max_count:
    #         print(f"Column '{col}' has missing values: {df_descr[col]['count']} out of {max_count}")
    
    total_rows = len(df)
    missing_counts = df.isna().sum()

    for col, missing in missing_counts.items():
        if missing > 0:
            print(f"Column '{col}' has {missing} missing values ({total_rows - missing} out of {total_rows} non-null).")
    print()
    nan_columns = get_nan_values(df)

    if len(nan_columns) > 0:
        nan_df = df[df.isnull().any(axis=1)]
        nan_patient_id = nan_df[id_col].unique()
        display(nan_df)
        print()
        _ = check_duplicate_rows(df, column=id_col)
        return nan_patient_id
    else:
        _ = check_duplicate_rows(df, column=id_col)
        
def plot_hist(df, column, title = None, xlabel = None, y_label = "Frequency", stat = 'frequency', figsize=(10, 6), hue= None, multiple='dodge', bins = 30, kde=False):
    plt.figure(figsize=figsize)
    if hue is None:
        sns.histplot(df[column], bins=bins, stat=stat, kde=kde)
    else:
        sns.histplot(data=df, x = column, bins=bins, stat=stat, hue = hue, multiple=multiple, kde=kde)
    plt.title(title if title else f"Distribution of {column}")
    plt.xlabel(xlabel if xlabel else column)
    plt.ylabel(y_label)
    plt.show()

def plot_violin(df, column, title = None, xlabel = None, y_label = "Frequency", figsize=(10, 6), hue= 'gender'):
    plt.figure(figsize=figsize)
    sns.violinplot(data=df, x = column, hue=hue)
    plt.title(title if title else f"Distribution of {column}")
    plt.xlabel(xlabel if xlabel else column)
    plt.ylabel(y_label)
    plt.show()

def scatterplot(df, x_list, y, hue = None, title = None, xlabel = None, ylabel = None, figsize=(10, 6), savepath = None):
    n = len(x_list)
    n_cols = 2
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows), sharex=False, sharey=False)

    # Ensure axes is always a flat 1D array
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])

    handles, labels = None, None

    for i, x in enumerate(x_list):
        ax = axes[i]
        if hue:
            sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, legend='auto')
        else:
            sns.scatterplot(data=df, x=x, y=y, ax=ax)

        ax.set_title(f"{x} vs {y}")
        ax.set_xlabel(xlabel if xlabel else x)
        ax.set_ylabel(ylabel if ylabel else y)

        if hue and handles is None:
            handles, labels = ax.get_legend_handles_labels()

        if hue:
            ax.get_legend().remove()

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if hue and handles:
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    if title:
        fig.suptitle(title)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    if savepath is not None:
        plt.savefig(os.path.join(savepath, f"{y}_scores_hue_{hue}_scatterplot.png"), bbox_inches='tight')
    plt.show()

def barplots(df, y_list, x, hue=None, title=None, xlabel=None, ylabel=None, figsize=(6, 4), savepath=None, order=None):
    n = len(y_list)
    n_cols = 2
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows), sharex=False, sharey=False)
    
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])

    handles, labels = None, None

    for i, y in enumerate(y_list):
        ax = axes[i]
        if hue:
            sns.barplot(
                data=df, x=x, y=y, hue=hue, ax=ax, legend=False,
                order=order if order is not None else sorted(df[x].dropna().unique())
            )
        else: 
            sns.barplot(
                data=df, x=x, y=y, ax=ax,
                order=order if order is not None else sorted(df[x].dropna().unique())
            )
        
        ax.set_title(f"{y} by {x}", fontsize=12)
        ax.set_xlabel(xlabel if xlabel else x, fontsize=10)
        ax.set_ylabel(ylabel if ylabel else y, fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        if hue and handles is None:
            handles, labels = ax.get_legend_handles_labels()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if hue and handles:
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    if savepath is not None:
        plt.savefig(os.path.join(savepath, f"{x}_scores_hue_{hue}_barplots.png"), bbox_inches='tight')
    plt.show()
    
def missing_from_df(df1, df2, df1_id, df2_id):
    df1_clean = df1[df1[df1_id].notna()]
    df2_clean = df2[df2[df2_id].notna()]
    
    missing_from_df1 = df2_clean[~df2_clean[df2_id].isin(df1_clean[df1_id])]
    missing_from_df2 = df1_clean[~df1_clean[df1_id].isin(df2_clean[df2_id])]
    
    return missing_from_df1, missing_from_df2


def check_img_resp_cluster_klscore(df, cluster_label, klscore, img_path,cluster_label_col = 'cluster_label', klscore_col = 'KL-Score', k = None):
    tocheck = df[(df[cluster_label_col]==cluster_label) & (df[klscore_col]==klscore)]
    idtocheck = list(tocheck['id'])
    to_remove = ["IM3003_1_left", "IM3012_2_right"]
    for i in to_remove:
        if i in idtocheck:
            idtocheck.remove(i)
    subfolders = ['test', 'train']
    l = []
    for folder in subfolders:
        basedir = os.path.join(img_path, folder, str(int(klscore)))
        possible_paths=[]
        for id in idtocheck:
            possible_path = id + '.png'
            possible_paths.append(possible_path)
        l_dir = os.listdir(basedir)
        for path in possible_paths:
            if path in l_dir:
                l.append(os.path.join(basedir, path))
            else:
                parent_path = os.path.join(img_path, os.path.basename(path))
                l.append(parent_path)
        break
    if k is not None:
        l = l[:k]

    cols = 2
    rows = math.ceil(len(l) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for i, p in enumerate(l):
        r, c = divmod(i, cols)
        img = load_image(p)  # assumes your existing helper
        axes[r, c].imshow(img, cmap='gray')
        axes[r, c].set_title(os.path.basename(p), fontsize=9)
        axes[r, c].axis('off')

    # Hide any unused axes (e.g., when len(paths) is odd)
    for j in range(len(l), rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis('off')

    plt.tight_layout()
    plt.show()
    return idtocheck

def boxplot(
    df: pd.DataFrame,
    y_list: list,
    x: str,
    hue: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize_per_panel=(5.5, 4.0),
    n_cols: int | None = None,
    order: list | None = None,
    hue_order: list | None = None,
    sharex: bool = False,
    sharey: bool = False,
    show_points: bool = True,
    points_alpha: float = 0.35,
    rotate_xticks: int = 30,
    showfliers: bool = False,
    whis: tuple | float = (5, 95),
    tight_rect=(0, 0, 0.92, 0.95),
    savepath: str | None = None,
    filename: str | None = None,
):
    # --- Category order handling (respects CategoricalDtype if present) ---
    if order is None:
        if pd.api.types.is_categorical_dtype(df[x]):
            order = list(df[x].cat.categories)
            order.sort()  # sort alphanumerically within categorical levels
        else:
            order = list(pd.unique(df[x].dropna()))
            order.sort()  # sort alphanumerically if not categorical
    if hue is not None and hue_order is None:
        if pd.api.types.is_categorical_dtype(df[hue]):
            hue_order = list(df[hue].cat.categories)
            hue_order.sort()  # sort alphanumerically within categorical levels
        else:
            hue_order = list(pd.unique(df[hue].dropna()))
            hue_order.sort()  # sort alphanumerically if not categorical

    # --- Grid geometry ---
    n = len(y_list)
    if n_cols is None:
        n_cols = 2 if n <= 4 else 3  # sensible default
    n_rows = math.ceil(n / n_cols)
    fig_w = figsize_per_panel[0] * n_cols
    fig_h = figsize_per_panel[1] * n_rows

    # --- Style (lightweight, readable) ---
    sns.set_context("talk")
    sns.set_style("whitegrid", {"axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.35})

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h), sharex=sharex, sharey=sharey,
        constrained_layout=False
    )
    axes = np.atleast_1d(axes).ravel()

    legend_handles, legend_labels = None, None

    for i, y in enumerate(y_list):
        ax = axes[i]

        # Boxplot
        sns.boxplot(
            data=df, x=x, y=y, hue=hue, order=order, hue_order=hue_order,
            ax=ax, dodge=True, showfliers=showfliers, whis=whis
        )

        # Optional jittered points overlay (helps see sample size & spread)
        if show_points:
            # stripplot is faster / less overplotty than swarm for big n
            sns.stripplot(
                data=df, x=x, y=y, hue=hue, order=order, hue_order=hue_order,
                ax=ax, dodge=True if hue else False, alpha=points_alpha, jitter=0.18,
                linewidth=0
            )

        # Collect legend once (we'll add a single figure legend)
        if hue and legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        # Clean up duplicate legends in each subplot
        if hue:
            ax.legend_.remove()

        # Labels & ticks
        ax.set_title(f"{y} by {x}", fontsize=12)
        ax.set_xlabel(xlabel if xlabel else x, fontsize=10)
        ax.set_ylabel(ylabel if ylabel else y, fontsize=10)
        ax.tick_params(axis="x", rotation=rotate_xticks)

        # A bit of visual polish
        sns.despine(ax=ax, left=False, bottom=False)

    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Global title
    if title:
        fig.suptitle(title, fontsize=14)

    # Single shared legend (if hue)
    if hue and legend_handles:
        fig.legend(legend_handles[:len(hue_order) if hue_order else None],
                   legend_labels[:len(hue_order) if hue_order else None],
                   loc="center left", bbox_to_anchor=(0.99, 0.5), frameon=False, title=hue)

    plt.tight_layout(rect=tight_rect)

    # Saving
    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        if filename is None:
            base = f"box_{x}_vs_{len(y_list)}y"
            if hue:
                base += f"_by_{hue}"
            filename = base + ".png"
        fig.savefig(os.path.join(savepath, filename), dpi=160, bbox_inches="tight")

    return fig, axes

def significant_heatmap(result, alpha = 0.05):
    mask = result < alpha
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(result, annot=True, fmt=".3f", cmap='coolwarm_r', cbar = False)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if not mask[i, j]:
                ax.text(j + 0.5, i + 0.5, '', ha='center', va='center', color='blue')
    plt.show()

def get_pvalues(df):
    pval = df.where(np.triu(np.ones(df.shape), k=1).astype(bool)).stack().values
    _, global_p = combine_pvalues(pval, method='stouffer')
    return pval, global_p

def kruskal_wallis_analysis(df, val_column, cluster_col):
    groups = [group[val_column].values for name, group in df.groupby(cluster_col)]
    stat, p = kruskal(*groups)
    print(f"Kruskal–Wallis H-statistic: {stat:.3f}")
    print(f"p-value: {p:.4f}")

    if stat > 10 and p < 0.05:
        print("Post-hoc Dunn's test results:")
        dunns = sp.posthoc_dunn(df, val_col=val_column, group_col=cluster_col, p_adjust='bonferroni')
        print(f"Combined p-value (Stouffer's method): {get_pvalues(dunns)[1]:.4f}")
        significant_heatmap(dunns.values)
        # display(sp.posthoc_dunn(df, val_col=val_column, group_col=cluster_col, p_adjust='bonferroni'))
        print("Post-hoc Conover's test results:")
        conover = sp.posthoc_conover(df, val_col=val_column, group_col=cluster_col, p_adjust='holm')
        print(f"Combined p-value (Stouffer's method): {get_pvalues(conover)[1]:.4f}")
        # display(sp.posthoc_conover(df, val_col=val_column, group_col=cluster_col, p_adjust='holm'))
        significant_heatmap(conover.values)

def get_metrics(df, label = "cluster_label", score = "mean"):
    results = {}
    labels = df[label].unique().tolist()
    labels.sort()

    res = spearmanr(df[score].tolist(), df[label].tolist())
    results['spearmanr'] = res[0]

    for i in range(len(labels)-1):
        df['binary_label'] = 0
        df.loc[df[label] > labels[i], 'binary_label'] = 1
        fpr, tpr, thresholds = roc_curve(np.array(df['binary_label']),np.array(df[score]))
        auc = auc(fpr, tpr)
        ap_score = average_precision_score(np.array(df['binary_label']),np.array(df[score]))
       
        results[f'cluster_{labels[i]}'] = {'roc_auc': auc,
                                             'average_precision': ap_score}
        # results[f'cluster_{labels[i]}'] = {'roc_auc': auc,
        #                                    'fpr': fpr,
        #                                    'tpr': tpr}
    return results
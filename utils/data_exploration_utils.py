import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import math
from utils.load_utils import load_image

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
    l = []
    for dirp, dirn, _ in os.walk(img_path):
        for folder in dirn:
            basedir = os.path.join(dirp, folder, str(klscore))
            possible_paths=[]
            for id in idtocheck:
                possible_path = id + '.png'
                possible_paths.append(possible_path)
            l_dir = os.listdir(basedir)
            for path in possible_paths:
                if path in l_dir:
                    l.append(os.path.join(basedir, path))
            break
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
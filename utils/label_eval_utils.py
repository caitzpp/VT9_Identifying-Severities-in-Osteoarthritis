import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd

def scatter_plot(df, x = 'label', x_title = 'KL Score', y = 'anoms_count', y_title = 'Count Anomalities', label = None, title = None, save_path = None): # y = 'anom_av', y_title = 'Average Anom Score'
    '''
    Get a scatter plot between the KL score and anom_count (so how often the sample is considered to be an anomality) OR the anom score
    '''
    plt.figure(figsize=(8, 6))
    if label is not None:
        plt.scatter(df[x], df[y], alpha=0.6, c = df[label], cmap='viridis')
        plt.colorbar(label=label)
    else:
        plt.scatter(df[x], df[y], alpha=0.6, color='blue')
    if title is not None:
       plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.grid(True)
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)

def boxplot(df, x = 'label', x_title = 'KL Score', y = 'anoms_count', y_title = 'Count Anomalities', title = None, save_path = None):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=x, y=y, data=df)
    if title is not None:
        plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)

def pairplot(df, values = ['label', 'anoms_count', 'sim', 'av'], hue = None, title = None, save_path = None):
    if hue is not None:
        sns.pairplot(df[values], hue=hue, palette='viridis')
    sns.pairplot(df[values])
    if title is not None:
        plt.title(title)
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)

def plot_comparison_boxplot(df1, df2, column, grouping_column = None,  title = None, log= False, save_path = None):
    if grouping_column is not None:
        df1 = df1[[grouping_column, column]].value_counts().reset_index()
        df2 = df2[[grouping_column, column]].value_counts().reset_index()

    else:
        df1 = df1[column].value_counts().reset_index()
        df2 = df2[column].value_counts().reset_index()

    df1["test_train"] = "Train"
    df2["test_train"] = "Test"

    if log == True:
        y_col = "log_count"
    else:
        y_col = "count"
    
    if grouping_column is not None:
        comp_df = pd.concat([df1[[grouping_column, column, 'test_train', "count"]], df2[[grouping_column, column, 'test_train', "count"]]])
        comp_df = comp_df.sort_values(by=column)
        if log == True:
            comp_df[y_col] = np.log(comp_df['count'])

        plt.figure(figsize=(10, 6))
        sns.barplot(x=column, y = y_col, hue=grouping_column, data=comp_df)
        if title is not None:
            plt.title(title)
        plt.xlabel('Value')
        plt.ylabel(y_col)
        plt.legend(title=grouping_column)
        plt.show()
    else:
        comp_df = pd.concat([df1[[column, 'test_train', "count"]], df2[[column, 'test_train', "count"]]])
        comp_df = comp_df.sort_values(by=column)
        if log == True:
            comp_df[y_col] = np.log(comp_df['count'])
            
        plt.figure(figsize=(10, 6))
        sns.barplot(x=column, y = y_col, hue="test_train", data=comp_df)
        if title is not None:
            plt.title(title)
        plt.xlabel('Value')
        plt.ylabel(y_col)
        plt.legend(title="test_train")
        plt.show()
    
    if save_path is not None:
        plt.savefig(save_path)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

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
        plt.title('Box Plot: Distribution of Anoms Count by Label')
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
        plt.title('Box Plot: Distribution of Anoms Count by Label')
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
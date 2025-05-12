import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def scatter_plot(df, save_path, x = 'label', x_title = 'KL Score', y = 'anoms_count', y_title = 'Count Anomalities'): # y = 'anom_av', y_title = 'Average Anom Score'
    '''
    Get a scatter plot between the KL score and anom_count (so how often the sample is considered to be an anomality) OR the anom score
    '''
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x], df[y], alpha=0.6, color='blue')
    #plt.title('Scatter Plot: Correlation between Label and Anoms Count')
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "test_image.png"))

def boxplot(df, x = 'label', x_title = 'KL Score', y = 'anoms_count', y_title = 'Count Anomalities'):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=x, y=y, data=df)
    #plt.title('Box Plot: Distribution of Anoms Count by Label')
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()

def pairplot(df, values = ['label', 'anoms_count', 'sim', 'av']):
    sns.pairplot(df[values])
    plt.show()
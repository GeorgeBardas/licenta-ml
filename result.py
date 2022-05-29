import pickle
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def create_results_df():
    """
    Generare root mean square error, mean absolute error, R2 score pentru fiecare model
    """
    results_dict = pickle.load(open("/Users/georgebardas/Documents/Projects/python/licenta-ml/model_scores.p", "rb"))

    results_df = pd.DataFrame.from_dict(results_dict, orient='index',
                                        columns=['RMSE', 'MAE', 'R2'])

    results_df = results_df.sort_values(by='RMSE',
                                        ascending=False).reset_index()

    results_df.to_csv('/Users/georgebardas/Documents/Projects/python/licenta-ml/data/results.csv')

    return results_df


def plot_results(results_df):
    """
    Generare grafic cu scor RMSE pentru fiecare model
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(np.arange(len(results_df)), 'RMSE', data=results_df, ax=ax,
                 label='RMSE', color='mediumblue')
    sns.lineplot(np.arange(len(results_df)), 'MAE', data=results_df, ax=ax,
                 label='MAE', color='Cyan')

    plt.xticks(np.arange(len(results_df)), rotation=45)
    ax.set_xticklabels(results_df['index'])
    ax.set(xlabel="Model",
           ylabel="Scor",
           title="Compararea erorilor modelelor")
    sns.despine()

    plt.savefig(f'/Users/georgebardas/Documents/Projects/python/licenta-ml/model_output/compare_models.png')


def main():
    results = create_results_df()
    plot_results(results)


main()

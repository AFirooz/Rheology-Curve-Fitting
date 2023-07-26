from os.path import join
from collections import namedtuple
import pandas as pd
import seaborn as sns
from glob import glob
import re
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = join('..', 'data')


def get_color_palette(y: list = None) -> sns.color_palette:
    if y is None:
        y = []

    snsc = namedtuple('snsc', ['red', 'blue', 'green', 'purple'])
    snsc = snsc(**{'red': '#e77c8d',
                   'blue': '#5ea5c5',
                   'green': '#56ad74',
                   'purple': '#a291e1'})

    diff_ = len(y) - len(snsc)

    if diff_ > 0:
        black = '#1f1f1f'
        snsc = snsc + (black,) * diff_

    return sns.color_palette(snsc)


def load_files_df(interested_in: [str], chosen_files: [int] = None, DATA_PATH: str = DATA_PATH) -> (
        [pd.DataFrame], str):
    # Using regex to find all the files that start with "pc" and end with ".xls"
    pattern = re.compile(r'^pc.*\.xls$', re.IGNORECASE)
    files = [f for f in glob('*', root_dir=DATA_PATH) if pattern.match(f)]

    if chosen_files is not None:
        files = [files[i] for i in chosen_files]

    if files:
        print('Found Files:')
        for i, file in enumerate(files):
            print(f'{i + 1} - "{file}"')

        # common_name is used when saving files to lower mistakes
        try:
            # name starts with "pc" and ends with "(number)mgml"
            common_name = re.findall(r'pc.*\(\d{1,3}mgml\)', files[0], re.IGNORECASE)[0].replace(' ', '_').lower()
        except IndexError:
            try:
                # name starts with "pc" and ends with "(number)m(number).(number)"
                common_name = re.findall(r'pc.*\d{1,3}m\d{1,3}\.\d{1,3}', files[0], re.IGNORECASE)[0].replace(' ',
                                                                                                              '_').lower()
            except IndexError:
                # name starts with "pc" and ends with "(number)m(number)"
                common_name = re.findall(r'pc.*\d{1,3}m\d{1,3}', files[0], re.IGNORECASE)[0].replace(' ', '_').lower()

        print(f"\nFiles related to -> {common_name}")

        na_values = ("NA", "N/A", "na", "n/a", "NULL", "null", "Not documented", "Not Documented", 'nan', '')

        # opening the files as dataframes
        dfs = [pd.read_excel(f"{DATA_PATH}/{file}",
                             index_col=None,
                             header=1,
                             sheet_name=1,
                             usecols=interested_in,
                             na_values=na_values).drop(labels=0, axis=0).reset_index(drop=True)
               for file in files]

        return dfs, files, common_name
    else:
        raise Exception('No files were found !')


def plot_xy(data: pd.DataFrame, x: str, y: [str], ax=None, fig=None, sep_plots: bool = False, save_str: str = None,
            scatter: bool = False, error: bool = False, legend: bool = False, log: bool = False) -> None:
    """
    To plot the data in a lineplot and scatterplot
    :param data:
    :param x:
    :param y:
    :param ax:
    :param fig:
    :param sep_plots: To plot each y in a separate plot or all in one plot
    :param save_str: The name of the file to save the plot, if None, the plot will not be saved
    :param scatter:
    :param error:
    :param legend:
    :return: None
    """
    if not sep_plots and log:
        ax.set_yscale('log')

    for col in y:
        if sep_plots:
            fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
            ax.set_yscale('log') if log else None
        if len(y) > 1:
            x = f"{x}_{col}"
        if 'file' in data.columns:
            sns.lineplot(data=data, x=x, y=col, ax=ax, hue='file')
            sns.scatterplot(data=data, x=x, y=col, ax=ax, hue='file') if scatter else None
        else:
            sns.lineplot(data=data, x=x, y=col, ax=ax)
            sns.scatterplot(data=data, x=x, y=col, ax=ax) if scatter else None
            ax.errorbar(data=data, x=x, y=col, yerr=f'{col}_stderr', errorevery=1, elinewidth=1,
                        alpha=0.3) if error else None

    ax.legend().set_visible(legend)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, fontsize="8") if legend else None
    plt.tight_layout()
    fig.savefig(join(DATA_PATH, f"{save_str}.png"), dpi=600) if save_str is not None else None
    plt.show()


def av_stderr(df: pd.DataFrame, bins: pd.IntervalIndex, x: str, y: [str], rename_x: str = None,
              save_str: str = None, DATA_PATH=DATA_PATH) -> pd.DataFrame:
    """

    :param save_str: If provided, the file will be saved as an Excel.
    :param y:
    :param x:
    :param rename_x: If a string provided, the x_av column will be renamed.
    :param bins: The bins that will be used in averaging.
    :param df: The merged data frame containing all vales that need to be averaged.
    :return:
    """
    df = df.copy()

    # Calculating the averaged x based on the bins provided
    df['x_av'] = pd.cut(df[x], bins=bins, include_lowest=True)
    df.reset_index(drop=True, inplace=True)
    # will use the average point of each bin as the new x
    df['x_av'] = df['x_av'].map(lambda x: ((x.left + x.right) / 2))

    # Calculating the averaged y and it's standard error
    dfs = []
    for col in y:
        # getting the mean values for all values in each bin
        pre_df = df.copy().groupby('x_av')[col]
        av_df = pre_df.mean()

        # getting the std values for all values in each bin
        std_df = pre_df.std()
        sqrt_len = np.sqrt(pre_df.count())
        stderr_df = (std_df / sqrt_len).round(5)
        stderr_df.fillna(value=0, inplace=True)  # todo: should we do this?
        stderr_df.rename(f'{col}_stderr', inplace=True)

        # adding the error (stderr) values to the averages
        col_df = pd.merge(av_df, stderr_df, how='outer', on='x_av', suffixes=(False, False))

        # processing the data and saving them
        col_df.dropna(subset=[col], inplace=True)
        col_df.reset_index(inplace=True)
        if rename_x is None:
            col_df.rename(columns={'x_av': f'x_av_{col}'}, inplace=True) if len(y) > 1 else None
        else:
            assert type(rename_x) == str, 'You need to provide a string for "rename_x"'
            col_df.rename(columns={'x_av': f'{rename_x}_{col}'}, inplace=True) if len(y) > 1 else \
                col_df.rename(columns={'x_av': f'{rename_x}'}, inplace=True)
        dfs.append(col_df)

    # putting everything together
    if len(y) > 1:
        final_df = pd.concat(dfs, axis=1, ignore_index=True)
    else:
        final_df = col_df

    if final_df.isna().sum().any() > 0:
        raise ValueError("You have NAs in final df, please double check!")

    final_df.to_excel(join(DATA_PATH, save_str), index=False) if save_str is not None else None
    return final_df


def assert_ratios(rat1, rat2):
    assert rat1 < 1, f'The {rat1} needs to be less than 1'
    assert rat2 < 1, f'The {rat2} needs to be less than 1'
    # assert rat1 + rat2 <= 1, f'The addition of {rat1} and {rat2} needs to be less than or equal to 1'
    assert rat1 < rat2, f'The {rat1} needs to be less than {rat2}'


def merge_dfs(dfs: [pd.DataFrame], files: [str]) -> pd.DataFrame:
    df_merged = dfs.copy()
    for i, name in enumerate(files):
        df_merged[i] = np.abs(df_merged[i])
        df_merged[i]['file'] = name

    return pd.concat(df_merged)

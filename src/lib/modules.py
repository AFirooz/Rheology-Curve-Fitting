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
    To plot the data in a line-plot and scatterplot,
    :param log:
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
    data = data.copy()
    nrows = len(y) if sep_plots else 1
    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 5 * nrows), sharex=True
                           ) if ax is None or fig is None else None
    ax = np.ravel(ax)

    for j, col in enumerate(y):
        i = j if sep_plots else 0
        ax[i].set_yscale('log') if log else None
        # setting the grid lines to only show along the x-axis
        ax[i].grid(visible=False, which='both')
        ax[i].grid(visible=True, which='major')
        ax[i].grid(visible=True, which='both', axis='y', alpha=0.5)

        if 'file' in data.columns:
            sns.lineplot(data=data, x=x, y=col, ax=ax[i], hue='file')
            sns.scatterplot(data=data, x=x, y=col, ax=ax[i], hue='file') if scatter else None
        else:
            xcol = f"{x}_{col}"
            sns.lineplot(data=data, x=xcol, y=col, ax=ax[i])
            sns.scatterplot(data=data, x=xcol, y=col, ax=ax[i]) if scatter else None
            if error:
                data['plty1'] = data[col] - data[f"{col}_stderr"]
                data['plty2'] = data[col] + data[f"{col}_stderr"]
                ax[i].fill_between(data=data, x=xcol, y1='plty1', y2='plty2', alpha=0.25)

        if legend:
            ax[i].legend().set_visible(True)
            ax[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, fontsize="small",
                         title=None)

    ax[0].set_ylabel(', '.join(y)) if 'file' in data.columns and not sep_plots else None
    plt.tight_layout()
    fig.savefig(join(DATA_PATH, f"{save_str}.png"), dpi=600) if save_str is not None else None
    plt.show()


def av_stderr(df: pd.DataFrame, bins: pd.IntervalIndex, x: str, y: [str], rename_x: str = None,
              save_str: str = None, DATA_PATH=DATA_PATH) -> pd.DataFrame:
    """
    For more information on calculating the error, read this: https://seaborn.pydata.org/tutorial/error_bars.html

    :param save_str: If provided, the file will be saved as an Excel.
    :param y:
    :param x:
    :param rename_x: If a string is provided, the x_av column will be renamed.
    :param bins: The bins that will be used in averaging.
    :param df: The merged data frame contains all vales that need to be averaged.
    :return:
    """

    df = df.copy()

    # Calculating the averaged x based on the bins provided
    df['x_av'] = pd.cut(df[x], bins=bins, include_lowest=True)
    df.reset_index(drop=True, inplace=True)
    # will use the average point of each bin as the new x
    df['x_av'] = df['x_av'].map(lambda z: round((z.left + z.right) / 2, 4))

    # Calculating the averaged y and its standard error
    dfs = []
    for col in y:
        # getting the mean values for all values in each bin
        pre_df = df.copy().groupby('x_av')[col]
        av_df = pre_df.mean()

        # getting the std values for all values in each bin
        std_df = pre_df.std()
        sqrt_len = np.sqrt(pre_df.count())
        stderr_df = (std_df / sqrt_len).round(5)
        stderr_df.fillna(method='ffill', inplace=True)  # there is another method called 'bfill'
        stderr_df.rename(f'{col}_stderr', inplace=True)

        # adding the error (stderr) values to the averages
        col_df = pd.merge(av_df, stderr_df, how='outer', on='x_av', suffixes=(False, False))

        # processing the data and saving them
        col_df.dropna(subset=[f"{col}"], inplace=True)
        # TODO: test and see what happens if two of the x-axis are different (one is NA and got dropped).
        #   Then concat the x-axis and keep it as index. Find out if it will create problems
        col_df.reset_index(inplace=True)  # we are keeping the old index because it was the 'x_av' column.
        if rename_x is None:
            col_df.rename(columns={'x_av': f'{x}_{col}'}, inplace=True) if len(y) > 1 else \
                col_df.rename(columns={'x_av': f'{x}'}, inplace=True)
        else:
            assert isinstance(rename_x, str), 'You need to provide a string for "rename_x"'
            col_df.rename(columns={'x_av': f'{rename_x}_{col}'}, inplace=True) if len(y) > 1 else \
                col_df.rename(columns={'x_av': f'{rename_x}'}, inplace=True)

        dfs.append(col_df)

    # putting everything together
    if len(y) > 1:
        # final_df = pd.concat(dfs, axis=1, ignore_index=True)  # This ignores column names too!
        final_df = pd.concat(dfs, axis=1)
        final_df.reset_index(drop=True, inplace=True)

    else:
        final_df = dfs[0]

    if final_df.isna().sum().any() > 0:
        raise ValueError("You have NAs in final df, please double check!")

    final_df.to_excel(join(DATA_PATH, save_str), index=False) if save_str is not None else None
    return final_df


def assert_ratios(rat1, rat2):
    assert rat1 < 1, f'The {rat1} needs to be less than 1'
    assert rat2 < 1, f'The {rat2} needs to be less than 1'
    # assert rat1 + rat2 <= 1, f'The addition of {rat1}, and {rat2} needs to be less than or equal to 1'
    assert rat1 < rat2, f'The {rat1} needs to be less than {rat2}'


def merge_dfs(dfs: [pd.DataFrame], files: [str]) -> pd.DataFrame:
    df_merged = dfs.copy()
    for i, name in enumerate(files):
        df_merged[i] = np.abs(df_merged[i])
        df_merged[i]['file'] = name

    return pd.concat(df_merged)


def create_bins(data: pd.DataFrame, x: str, step: [int] or int,
                ratios: [float] = None) -> pd.IntervalIndex or np.Arrayterator:
    """
    This is function is used to create the needed bins to be used with av_stderr().
    If the mask was assigned a fixed number, it will be treated as a noise mask and the bins will be equally spaced.
    Otherwise, if the mask was given, a list of 3 numbers will be treated as
    [init_smoothness, med_smoothness, end_smoothness]. Ratios refer to the location of the split location
    of the range to assign dynamic bins.

    Example:
        If the mask was given [25, 100, 100] and the ratios are [0.25, 0.75] then the bins will have more details
        (smaller step size) at the first 25% of the range, then for the next 50% of the range, there will be fewer bins
        and finally for the last 25% of the range, there will be even fewer bins.

    Note that, for now, you can't create dynamic bins with a step size smaller than 1.

    :param data: The DataFrame you will be working on
    :param x: The x-axis name
    :param ratios: A list of 2 numbers that relate to two splits in the bins.
    :param step:
    :return:
    """

    min_val = data[x].min() - 1
    max_val = data[x].max() + 1

    if ratios is None:
        ratios = [.25, .75]
    else:
        assert_ratios(ratios[0], ratios[1])

    if isinstance(step, list):
        print(f"Using dynamic bins")

        dist = max_val - min_val + 2
        s0 = float(min_val)
        s1 = float(dist * ratios[0] + s0)
        s2 = float(dist * ratios[1] + s0)
        s3 = float(max_val + 1)

        intervals = np.concatenate([
            np.arange(start=s0, stop=s1, step=min(step[0], s1 - s0)),
            np.arange(start=s1, stop=s2, step=min(step[1], s2 - s1)),
            np.arange(start=s2, stop=s3 + step[2], step=min(step[2], s3 - s2))
        ])
        bins = pd.IntervalIndex.from_breaks(intervals)

    else:
        print(f"Using fixed bins")
        assert isinstance(step, int), 'step needs to be an integer or a list'
        bins = np.arange(start=min_val, stop=max_val + step, step=step)

    print(f"Number of bins = {bins.shape[0]}")
    return bins

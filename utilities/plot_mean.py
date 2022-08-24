import argparse
import itertools
import os
import sys
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def method_label(filename: str):
    if filename.find("PQL") != -1:
        return "PQL"
    if filename.find("QLTT") != -1:
        return "QLTT"
    if filename.find("QLCO") != -1:
        return "QLCO"
    else:
        return "DUA"


def list_files(path: str) -> List[str]:
    stream = os.popen(f"ls {path}")
    output = stream.read()
    files = output.split('\n')
    files.pop()

    return [f"{path}/{file}" for file in files]


def set_label(label: str, count: int):
    """Sets plot label based on name

    Args:
        label (str): method name to label
        count (int): order of the name in plot
    """

    if label == "Occupancy":
        label += f" (%) [{chr(97 + count)}]"
    elif label == "Travel Time":
        label = f" Average Network Travel Time (s) [{chr(97 + count)}]"
    elif label == "CO":
        label = f" Total Netowrk CO Emission (mg) [{chr(97 + count)}]"
    elif label == "Speed":
        label = f" Average Network Speed (m/s) [{chr(97 + count)}]"
    elif label == "Running Vehicles":
        label += f" [{chr(97 + count)}]"

    return label


def gen_df(path: str,
           cut_step: int = 1500,
           plot_type: Union[List[str], None] = None,
           avg_gap: int = 600,
           compressed: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates a dataframe from csv file.

    Args:
        path (str): Path to csv file.
        cut_step (int, optional): Step to stop plotting. Defaults to 50000.
        plot_type (List[str], optional): Type of plot passed inside a list. Defaults to ["mean"] which means it will take an average from all links.
        avg_gap (int, optional): Window to take the moving average in plot. Defaults to 600.
        compressed (bool, optional): flag that indicates if file is compressed or not. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: _description_
    """
    plot_type = plot_type or ["mean"]
    print(f"Reading file {path} ...", end='', flush=True)
    df = pd.DataFrame(pd.read_csv(path, compression="xz") if compressed else pd.read_csv(path))
    print("done.")
    print("Processing ...", end='', flush=True)

    df = df.loc[(df["Step"] >= cut_step) & (df["Step"] <= 50000)]
    df.loc[df["Occupancy"] == 0, ["Travel Time", "Speed"]] = np.nan
    # df = df[["Step", "Link", "Travel Time", "CO", "Speed"]]
    df["Speed / Running Vehicles"] = df["Speed"] / (df["Running Vehicles"] + 1)
    if len(plot_type) > 1:
        df = df[df["Link"].isin(plot_type)]
        df.pop("Link")
    elif plot_type[0] == "old":
        pass

    df = df.groupby(["Step"]).agg({"Occupancy": "mean",
                                   "Travel Time": "mean",
                                   "CO": "mean",
                                   "Speed": "mean",
                                   "Running Vehicles": "sum",
                                   "Speed / Running Vehicles": "mean"}).reset_index()
    rolling_avg = df.copy(deep=True)
    for column in rolling_avg.columns[1:]:
        rolling_avg[column] = rolling_avg[column].rolling(avg_gap).mean()
    print("done.")

    rolling_avg["Method"] = pd.Series([method_label(path) for _ in range(len(rolling_avg.index))])
    return df, rolling_avg


def gen_plots(filepaths: List[str],
              cut_step: int = 1500,
              plot_type: Union[List[str], None] = None,
              avg_gap: int = 600,
              plot_style: str = "line",
              compressed: bool = True,
              multiple: bool = False,
              parameters: Union[None, List[str]] = None) -> pd.DataFrame:
    sns.set_theme(style="darkgrid")
    plot_type = plot_type or ["mean"]
    parameters = parameters or ["Step", "Method", "Travel Time", "CO", "Speed"]

    if multiple:
        size = len(filepaths)
        paths = [list_files(path) for path in filepaths]
        print("multiple")
    else:
        size = 1
        paths = [filepaths]

    palette = itertools.cycle(sns.color_palette("tab10", size))  # type: ignore
    plot = sns.scatterplot if plot_style == "scatter" else sns.lineplot

    full_df: Union[pd.DataFrame, None] = None
    columns = []
    for i in range(size):
        dfs: List[Tuple[pd.DataFrame, pd.DataFrame]] = [gen_df(file, cut_step, plot_type, avg_gap, compressed)
                                                        for file in paths[i]]
        columns = [col for col in dfs[0][0].columns[1:] if col in parameters]
        if i == 0:
            full_df = pd.DataFrame(dict({"Step": []}, **{obj: [] for obj in columns}))
        for _, rolling in dfs:
            full_df = pd.concat([full_df, rolling], ignore_index=True)

    fig, axes = plt.subplots(len(columns), 1, figsize=(15, len(columns) * 4.5), constrained_layout=True)
    plot_color = next(palette)
    if len(columns) > 1:
        for j, col in enumerate(columns):
            plot(x='Step', y=col, data=full_df[parameters], ax=axes[j],
                 ci='sd', legend="full", hue="Method")  # type: ignore
            axes[j].set_ylabel(set_label(col, j))  # type: ignore
    else:
        plot(x="Step", y=columns[0], data=full_df, ax=axes, ci='sd', color=next(palette))  # type: ignore

    # for ax in axes:
    #     ax.legend(legends)
    # fig.legend(legends)  # type: ignore
    if plot_type[0] == "mean":
        fig.suptitle("Methods Comparison - Average of 30 Runs - Full Network")
    else:
        fig.suptitle("Methods Comparison - Average of 30 Runs - Horizontal Links")
    plt.show()

    return full_df
    # return [df for df, _ in dfs]


def main():
    parser = argparse.ArgumentParser(
        description='Script to take several csv files of same sort and take the average of the values to store')

    parser.add_argument("-f", "--folder", action="store", dest="folder_path",
                        help="Path to folder containing csv files to take average (mandatory)")
    parser.add_argument("-m", "--multiple", action="store_true", dest="multiple", default=False,
                        help="Indicates if the means are from multiple simulations.")

    opt = parser.parse_args()
    if not opt.folder_path:
        print('Wrong usage of script!')
        print()
        parser.print_help()
        sys.exit()

    filenames = list_files(opt.folder_path)
    gen_plots(filenames, multiple=opt.multiple)


if __name__ == "__main__":
    main()

import argparse
import itertools
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    sns.set_theme(style="darkgrid")
    palette = itertools.cycle(sns.color_palette("colorblind"))  # type: ignore

    parser = argparse.ArgumentParser(
        description='Script to plot a scatter graph using a csv file with two columns')

    parser.add_argument("-f", "--file", action="store", dest="csv_file",
                        help="CSV file containing data (mandatory)")

    parser.add_argument("-t", "--type", action="store", nargs="+", dest="plot_type", default=["mean"],
                        help="Indicates what type of plot should be used (old, mean, or link [id]). Default = mean")

    parser.add_argument("-m", "--moving-avg-gap", action="store", dest="avg_gap", default=100, type=int,
                        help="Determines the gap to calculate the moving average. Default = 100 steps")

    parser.add_argument("-s", "--style", action="store", dest="plot_style", default="scatter",
                        help="Define plot style. Default = scatter")

    parser.add_argument("-c", "--cut-step", action="store", dest="cut_step", default=0, type=int,
                        help="Step to cut the dataframe and start the plot")

    parser.add_argument("--compressed", action="store_true", dest="compressed", default=False,
                        help="Indicates if it is a compressed file")

    options = parser.parse_args()
    if not options.csv_file:
        print('Wrong usage of script!')
        print()
        parser.print_help()
        sys.exit()

    df = pd.read_csv(options.csv_file, compression="xz") if options.compressed else pd.read_csv(options.csv_file)
    if options.plot_type != "old":
        n_links = df.groupby("Link")["Link"].nunique().size
        df = df.iloc[options.cut_step*n_links:]
    else:
        df = df.iloc[options.cut_step/100:]

    if options.plot_type[0] == "mean":
        df = df.groupby(["Step"]).agg("mean").reset_index()
        for column in df.columns:
            df[column] = df[column].rolling(options.avg_gap).mean()
    elif options.plot_type[0] == "old":
        pass
    else:
        df = df[df["Link"].isin(options.plot_type)]
        df.pop("Link")
        df = df.groupby(["Step"]).agg("mean").reset_index()
        for column in df.columns:
            df[column] = df[column].rolling(options.avg_gap).mean()

    columns = list(df.columns[1:])
    _, axes = plt.subplots(len(columns), 1, figsize=(10, len(columns) * 3.5), sharex=True, constrained_layout=True)
    plot = sns.scatterplot if options.plot_style == "scatter" else sns.lineplot
    if len(columns) > 1:
        for i, col in enumerate(columns):
            # axes[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plot(x="Step", y=col, data=df, ax=axes[i], color=next(palette))  # type: ignore
    else:
        # axes.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plot(x="Step", y=columns[0], data=df, ax=axes, color=next(palette))

    plt.show()

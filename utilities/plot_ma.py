import sys
import argparse
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    sns.set_theme(style="darkgrid")
    palette = itertools.cycle(sns.color_palette("colorblind"))

    parser = argparse.ArgumentParser(
        description='Script to plot a scatter graph using a csv file with two columns')

    parser.add_argument("-f", "--file", action="store", dest="csv_file",
                        help="CSV file containing data (mandatory)")

    options = parser.parse_args()
    if not options.csv_file:
        print ('Wrong usage of script!')
        print ()
        parser.print_help()
        sys.exit()

    df = pd.read_csv(options.csv_file)
    columns = list(df.columns[1:])
    fig, axes = plt.subplots(len(columns), 1, figsize=(10, len(columns) * 3.5), sharex=True, constrained_layout=True)
    if len(columns) > 1:
        for i, col in enumerate(columns):
            axes[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            sns.scatterplot(x="Step", y=col, data=df, ax=axes[i], color=next(palette))
    else:
        axes.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        sns.scatterplot(x="Step", y=columns[0], data=df, ax=axes, color=next(palette))

    plt.show()

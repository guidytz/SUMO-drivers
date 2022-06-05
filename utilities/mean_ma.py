import os
import sys
import argparse
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    sns.set_theme(style="darkgrid")
    palette = itertools.cycle(sns.color_palette("colorblind"))

    parser = argparse.ArgumentParser(
        description='Script to take several csv files of same sort and take the average of the values to store')

    parser.add_argument("-f", "--folder", action="store", dest="folder_path",
                        help="Path to folder containing csv files to take average (mandatory)")

    args = parser.parse_args()
    if not args.folder_path:
        print ('Wrong usage of script!')
        print ()
        parser.print_help()
        sys.exit()

    stream = os.popen(f"ls {args.folder_path}")
    output = stream.read()
    files = output.split('\n')
    files.pop()

    objectives = list(pd.read_csv(f"{args.folder_path}/{files[0]}").columns[1:])
    full_df = pd.DataFrame(dict({"Step": []}, **{obj: [] for obj in objectives}))
    for file in files:
        csv_df = pd.read_csv(f"{args.folder_path}/{file}")
        csv_df = csv_df.iloc[50:]
        full_df = full_df.append(csv_df, ignore_index=True)

    _, axes = plt.subplots(len(objectives), 1, figsize=(10, len(objectives) * 3.5), 
                            sharex=True, constrained_layout=True)
    if len(objectives) > 1:
        for i, obj in enumerate(objectives):
            axes[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            sns.lineplot(x='Step', y=obj, data=full_df, ax=axes[i], ci='sd', color=next(palette))
    else:
        axes.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        sns.lineplot(x='Step', y=objectives[0], data=full_df, ax=axes, ci='sd', color=next(palette))
    plt.show()


if __name__ == "__main__":
    main()

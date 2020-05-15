import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to plot a scatter graph using a csv file with two columns')

    parser.add_argument("-f1", "--file1", action="store", dest="csv_file_1",
                        help="CSV file containing data of first plot (mandatory)")
    parser.add_argument("-f2", "--file2", action="store", dest="csv_file_2",
                        help="CSV file containing data of second plot (mandatory)")

    options = parser.parse_args()
    if not options.csv_file_1 and not options.csv_file_2:
        print ('Wrong usage of script!')
        print ()
        parser.print_help()
        sys.exit()

    legend_1 = "SUMO DUA" if str(options.csv_file_1).find("not_learning") != -1 else "Q-Learning"
    legend_1 = f"{legend_1} with C2I" if str(options.csv_file_1).find("C2I") != -1 else f"{legend_1}"
    legend_2 = "SUMO DUA" if str(options.csv_file_2).find("not_learning") != -1 else "Q-Learning"
    legend_2 = f"{legend_2} with C2I" if str(options.csv_file_2).find("C2I") != -1 else f"{legend_2}"

    df1 = pd.read_csv(options.csv_file_1)
    df2 = pd.read_csv(options.csv_file_2)
    columns = list(df1.columns)
    ax = df1.plot(kind="scatter", x=columns[0], y=columns[1], s=5)
    df2.plot(kind="scatter", x=columns[0], y=columns[1], ax=ax, color="C1", s=5)
    plt.ylabel("Average travel time")
    ax.legend([legend_1, legend_2])
    plt.show()
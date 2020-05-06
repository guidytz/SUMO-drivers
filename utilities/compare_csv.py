import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to plot a scatter graph using a csv file with two columns')

    parser.add_argument("-f1", "--file1", action="store", dest="csv_file1",
                        help="CSV file containing data of first plot (mandatory)")
    parser.add_argument("-f2", "--file2", action="store", dest="csv_file2",
                        help="CSV file containing data of second plot (mandatory)")

    options = parser.parse_args()
    if not options.csv_file1 and not options.csv_file2:
        print ('Wrong usage of script!')
        print ()
        parser.print_help()
        sys.exit()

    df1 = pd.read_csv(options.csv_file1)
    df2 = pd.read_csv(options.csv_file2)
    columns = list(df1.columns)
    ax1 = df1.plot(kind="scatter", x=columns[0], y=columns[1], s=5)
    df2.plot(kind="scatter", x=columns[0], y=columns[1], ax=ax1, color="red", s=5)
    ax1.legend(["SUMO DUA", "Q-Learning"])
    plt.show()
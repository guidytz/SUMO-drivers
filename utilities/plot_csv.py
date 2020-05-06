import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
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
    columns = list(df.columns)
    df.plot(kind="scatter", x=columns[0], y=columns[1], s=5)
    plt.show()

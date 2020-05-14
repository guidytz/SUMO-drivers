import sys
import argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to plot a bar graph using a csv file with two columns')

    parser.add_argument("-f", "--file", action="store", dest="csv_file",
                        help="CSV file containing data of first file (mandatory)")

    options = parser.parse_args()
    if not options.csv_file:
        print ('Wrong usage of script!')
        print ()
        parser.print_help()
        sys.exit()

    df = pd.read_csv(options.csv_file)
    col = list(df.columns)
    df.plot(kind="bar", x=col[0], y=col[1], figsize=(15, 7))
    plt.ylabel("Number of trips ended")
    plt.subplots_adjust(left=0.07, bottom=0.20, right=0.95, top=0.95)
    plt.show()
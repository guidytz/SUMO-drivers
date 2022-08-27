import sys
import os
import pandas as pd

def get_all_csv_files_in_dict(file: str) -> list:
	files_list_dict = []

	path = file[:-1] # removes '.' from file
	
	files_list_dict = os.listdir(path)

	filtered_files_list = []
	for file in files_list_dict:
		if os.path.splitext(file)[1] == ".csv":
			file = path + file
			filtered_files_list.append(file)

	return filtered_files_list

def checks_lines_columns_files(files: list) -> bool:
	list_line_column_tuple = []
	for file in files:
		df = pd.read_csv(file)
		nlines = len(df.axes[0])
		ncolumns = len(df.axes[1])
		list_line_column_tuple.append((nlines, ncolumns))
	
	for i in range(1, len(list_line_column_tuple)):
		if list_line_column_tuple[i] != list_line_column_tuple[i-1]:
			return False
	
	return True

if __name__ == "__main__":
 
	'''
	Aggregates data of csv file by step, taking the mean average of the data and returning it in another csv.

	Input could either be a list of files or a directory ending in '.', in which case the program will get all .csv files from directory
	'''

	file_list = sys.argv[1:] 

	print("Processing file names...")
	processed_file_list = []
	for file in file_list:
		if file.endswith("."):
			processed_file_list += get_all_csv_files_in_dict(file)
		else:
			processed_file_list.append(file)

	print("Files: ")
	for filename in processed_file_list:
		print(filename) 

	if checks_lines_columns_files(processed_file_list):
		print("All files have the same number of lines and columns")
	else:
		print("Warning: not all files have the same number of lines and columns")
	
	for fname in processed_file_list:
		df=pd.read_csv(fname)
		dfaggr = df.groupby("Step").agg("mean").reset_index()
		fname_save = os.path.splitext(fname.split("/")[-1])[0] # gets name of file 
		dfaggr.to_csv('aggr_'+fname_save+'.csv', index=False)
		print('aggr_'+fname_save+'.csv created in current directory')

import os
from typing import List
import errno
from datetime import datetime
from random import SystemRandom
import numpy as np
import pandas as pd

class DataCollector:
    """Class responsible for collecting data from the simulation and saving it to csv files.

    Args:
        sim_filename (str): Name of the simulation for saving purposes
        steps_to_measure (int, optional): Steps to calculate moving average. Defaults to 100.
        custom_path (str, optional): Custom path to save the files. Defaults to ''.
        additional_folders (List[str], optional): additional folders to add in path to distingish simulations.
        Defaults to None.
        param_list (List[str], optional): list containing all measured params
    """

    def __init__(self, sim_filename: str = "default",
                 steps_to_measure: int = 100,
                 custom_path: str = '',
                 additional_folders: List[str] = None,
                 param_list: List[str] = None) -> None:
        if sim_filename == "default":
            print("Warning: using 'default' as simulation name since the data collector wasn't informed.")
            print("Results will be saved in a default folder and might not be distinguishable from other simulations")
        self.__sim_filename = sim_filename
        self.__steps_to_measure = steps_to_measure
        self.__path = custom_path if custom_path != '' else f"{os.getcwd()}/results"
        self.__additional_folders = additional_folders
        self.__param_list = param_list if "TravelTime" in param_list else ["TravelTime"] + param_list
        self.__param_list = [item.replace("TravelTime", "Travel Time") for item in self.__param_list]
        self.__data_collected = [np.array([]) for _ in self.__param_list]
        self.__data_df = pd.DataFrame(dict({"Step": []}, **{obj: [] for obj in self.__param_list}))
        self.__start_time = datetime.now()
        self.__random = SystemRandom()

    def append_data(self, data: List[List[int]], step: int) -> None:
        """Method that receives a list of travel times and the step they were retrieved and saves them to calculate the
        moving average.

        Args:
            data (List[List[int]]): list containing all travel times retrieved in the step given.
            step (int): step in which the travel times were retrieved.
        """
        for item in data:
            self.__data_collected = [np.append(param, val) for param, val in zip(self.__data_collected, item)]
        self.__update_data_df(step)

    def save(self):
        """Method that saves the data stored to csv file.
        """
        self.__save_to_csv("MovingAverage", self.__data_df)

    def reset(self) -> None:
        """Method that resets the collector data to make a new run.
        """
        self.__start_time = datetime.now()
        self.__data_collected = [np.array([]) for _ in self.__param_list]
        self.__data_df = pd.DataFrame(dict({"Step": []}, **{obj: [] for obj in self.__param_list}))
    
    def time_to_measure(self, step) -> bool:
        """Method that indicates if it's time to measure the moving average of the data collected based on the step 
        given when collector was created.

        Args:
            step (int): current step

        Returns:
            bool: returns a boolean that indicates if a measurement should be made.
        """
        return step % self.__steps_to_measure == 0

    def __update_data_df(self, step: int) -> None:
        """Method that updates the internal dataframe with the new moving average if the current step is the one to make
        the measurement.

        Args:
            step (int): current step
        """
        if self.time_to_measure(step) and self.__data_collected[0].size != 0:
            avg_data = [data.mean() for data in self.__data_collected]
            df_update = pd.DataFrame(dict({"Step": [step]},
                                        **{obj: [val] for obj, val in zip(self.__param_list, avg_data)}))
            self.__data_df = self.__data_df.append(df_update, ignore_index=True)
            self.__data_collected = [np.array([]) for _ in self.__param_list]

    def __create_folder(self, folder_name: str) -> None:
        """Method that creates a folder to save the files if it wasn't created yet.

        Args:
            folder_name (str): name/path to folder

        Raises:
            OSError: if the folder can't be created, it raises an OSError.
        """
        try:
            os.mkdir(folder_name)
        except OSError as error:
            if error.errno != errno.EEXIST:
                print(f"Couldn't create folder {folder_name}, error message: {error.strerror}")
                raise OSError(error).with_traceback(error.__traceback__)

    def __verify_and_create_folder_path(self, folder_name: str) -> str:
        """Method that creates the folder hierarchy where the simulation files will be stored. 

        Args:
            folder_name (str): Main simulation folder name

        Returns:
            str: Complete path to the final folder.
        """
        date_folder = self.__start_time.strftime("%m_%d_%y")
        folder_str = self.__path
        self.__create_folder(folder_str)
        folder_str += f"/{folder_name}"
        self.__create_folder(folder_str)
        folder_str += f"/{self.__sim_filename}"
        self.__create_folder(folder_str)
        folder_str += f"/{date_folder}"
        self.__create_folder(folder_str)

        if self.__additional_folders is not None:
            for additional_folder in self.__additional_folders:
                folder_str += f"/{additional_folder}"
                self.__create_folder(folder_str)

        return folder_str


    def __save_to_csv(self, folder_name: str, data_frame: pd.DataFrame) -> None:
        """Method that saves the data collected to a csv file.

        Args:
            folder_name (str): Main simulation folder name to save the file in.
            data_frame (pd.DataFrame): dataframe that stores the data to be saved.
        """
        folder_str = self.__verify_and_create_folder_path(folder_name)
        file_signature = f"{self.__start_time.strftime('%H-%M-%S')}_{self.__random.randint(0, 1000):03}"
        csv_filename = folder_str + f"/sim_{file_signature}.csv"
        data_frame.to_csv(csv_filename, index=False)


class ObjectiveCollector:

    def __init__(self, objecitve_list: List[str], sim_path: str) -> None:
        self.__objectives = objecitve_list
        self.__collector = pd.DataFrame({obj: [] for obj in self.__objectives})
        self.__sim_path = sim_path

    @property
    def __sim_path(self):
        return self.__existing_path

    @__sim_path.setter
    def __sim_path(self, path):
        if os.path.exists(path):
            self.__existing_path = path
        else:
            raise FileNotFoundError(f"The path `{path}` informed does not exist.")

    def append_rewards(self, reward_list: List[np.array]) -> None:
        reward_list = np.array(reward_list)
        n_obj = len(self.__objectives)
        new_data = pd.DataFrame({obj: reward_list[:, i] for obj, i in zip(self.__objectives, range(n_obj))})
        self.__collector = self.__collector.append(new_data, ignore_index=True)

    def save(self):
        filename = f"{self.__sim_path}/fit_data_{'_'.join(self.__objectives)}.csv"
        self.__collector.to_csv(filename, index=False, mode='a')

    def __str__(self) -> str:
        return f"{self.__collector}"

import os
from datetime import datetime
from pathlib import Path
from random import SystemRandom

import numpy as np
import pandas as pd


class DefaultCollector:
    """Default collector class that aggregates and saves data from the simulation.

    Args:
        aggregation_interval (int): interval that determines when the multiple data should be aggregated into the
        main dataframe.
        path (str): main folder path were the data file should be saved.
        params (List[str]): params that should be saved throughout the simulation. Note that the first param is the
        one considered the x axis within the dataframe, therefore this value will not be aggregated, but used as
        reference to the other params aggregated.
    """

    def __init__(self, aggregation_interval: int, path: Path, params: list[str]) -> None:
        self._start_time = datetime.now()
        self._aggregation_interval = aggregation_interval
        self._path = path
        self._params = params
        self._collector_df = self._empty_df[self._params[1:]]
        self._aggr_df = self._empty_df
        self._random = SystemRandom()
        self._name_addon = ""

    def append(self, data_dict: dict[str, list[int]]) -> None:
        """Method the appends data present in the dictionary to the collector dataframe.
        Note that the dictionary has to have the same keys passed as params when class was created.

        Args:
            data_dict (dict[str, list[int]]): data to append to the collector.
        """
        collector_data = {key: data_dict[key] for key in self._params[1:]}
        self._collector_df = pd.concat([self._collector_df, pd.DataFrame(collector_data)], ignore_index=True)
        curr_value = data_dict[self._params[0]][0]
        if self._should_aggregate(curr_value):
            self._aggregate(curr_value)

    def save(self) -> None:
        """Method that saves the data stored to csv file.
        """
        self._path.mkdir(exist_ok=True, parents=True)
        signature = Path(f"sim_{self._start_time.strftime('%H-%M-%S')}_{self._random.randint(0, 1000):03}.csv")
        csv_filename = self._path / signature
        self._aggr_df.to_csv(str(csv_filename), index=False)  # compression="xz"

    def reset(self) -> None:
        """Method that resets the collector data to make a new run.
        """
        self._collector_df = self._empty_df[self._params[1:]]
        self._aggr_df = self._empty_df
        self._start_time = datetime.now()

    @property
    def watched_params(self) -> list[str]:
        """Property that returns the params beeing watched in this run.

        Returns:
            list[str]: list of watched params.
        """
        return self._params

    def _aggregate(self, curr_value: int) -> None:
        """Method that aggregates the values of the collector dataframe into the main dataframe.
        Currently using the mean to aggregate.

        Args:
            curr_value (int): current value of the first parameter that is used as the x axis in the dataframe.
        """
        aggregated_df = self._collector_df.mean().to_frame().transpose()
        aggregated_df[self._params[0]] = [curr_value]
        self._update_main_dfs(aggregated_df)

    def _update_main_dfs(self, aggregated_df):
        """Method that updates main dfs when aggregation is done.

        Args:
            aggregated_df (pd.DataFrame): aggregated information to append to main df.
        """
        #print(f"{aggregated_df.columns=}\n")
        #print(f"{self._aggr_df.columns=}")
        self._aggr_df = pd.concat([self._aggr_df, aggregated_df[self._params]], ignore_index=True)
        self._collector_df = self._empty_df[self._params[1:]]

    def _should_aggregate(self, curr_value: int) -> bool:
        """Method that determines if the data collected should be aggregated in the main dataframe.

        Args:
            curr_value (int): current value of the first parameter that is used as the x axis in the dataframe.

        Returns:
            bool: flag indicating if it's time to aggregate the other params based on the interval informed.
        """
        return curr_value % self._aggregation_interval == 0

    @property
    def _empty_df(self) -> pd.DataFrame:
        """Property that returns an empty dataframe using the params as columns.

        Returns:
            pd.DataFrame: empty dataframe using the main params as columns.
        """
        return pd.DataFrame({param: [] for param in self._params})

    def __str__(self) -> str:
        return f"{self._aggr_df}"


class TripCollector(DefaultCollector):
    """Class responsible for collecting data from the simulation and saving it to csv files.

    Args:
        network_name (str): Name of the simulation for saving purposes
        aggregation_interval (int, optional): Steps to calculate moving average. Defaults to 100.
        custom_path (str, optional): Custom path to save the files. Defaults to ''.
        additional_folders (list[Path], optional): additional folders to add in path to distingish simulations.
        Defaults to None.
        param_list (list[str], optional): list containing all measured params
        date (datetime, optional): datetime object that stores the simulation begin time. Defaults to datetime.now().
    """

    def __init__(self, network_name: str = "default",
                 aggregation_interval: int = 100,
                 custom_path: str | None = None,
                 additional_folders: list[Path] | None = None,
                 param_list: list[str] | None = None,
                 date: datetime = datetime.now()) -> None:
        if network_name == "default":
            print("Warning: using 'default' as simulation name since the data collector wasn't informed.")
            print("Results will be saved in a default folder and might not be distinguishable from other simulations")

        date_str: str = f"/{date.strftime('%m_%d_%y')}"
        additional_paths = Path(*additional_folders) if additional_folders is not None else Path()
        path = Path(f"{(custom_path or 'results')}/TripMovingAverage/{network_name}/{date_str}") / additional_paths
        param_list = param_list or []
        params = param_list if "TravelTime" in param_list else ["TravelTime"] + param_list
        params = [item.replace("TravelTime", "Travel Time") for item in params]
        params = ["Step"] + params
        super().__init__(aggregation_interval, path, params)

    def append_list(self, data: list[list[int]], step: int) -> None:
        """Method that receives a list of travel times and the step they were retrieved and saves them to calculate the
        moving average.

        Args:
            data (list[list[int]]): list containing all travel times retrieved in the step given.
            step (int): step in which the travel times were retrieved.
        """
        obj_dict = {obj: [] for obj in self._params[1:]}
        for item in data:
            for obj, val in zip(self._params[1:], item):
                obj_dict[obj].append(val)
        data_dict = dict({"Step": [step]}, **obj_dict)
        super().append(data_dict)

    def time_to_measure(self, step) -> bool:
        """Method that indicates if it's time to measure the moving average of the data collected based on the step
        given when collector was created.

        Args:
            step (int): current step

        Returns:
            bool: returns a boolean that indicates if a measurement should be made.
        """
        return self._should_aggregate(step)


class ObjectiveCollector:
    def __init__(self, objective_list: list[str], sim_path: Path) -> None:
        self.__objectives = objective_list
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

    def append_rewards(self, reward_list: list[np.ndarray]) -> None:
        reward_arr = np.array(reward_list)
        n_obj = len(self.__objectives)
        new_data = pd.DataFrame({obj: reward_arr[:, i] for obj, i in zip(self.__objectives, range(n_obj))})
        self.__collector = pd.concat([self.__collector, new_data], ignore_index=True)

    def save(self):
        filename = Path(f"{self.__sim_path}/fit_data_{'_'.join(self.__objectives)}.csv")
        self.__collector.to_csv(str(filename), index=False, mode='a')

    def __str__(self) -> str:
        return f"{self.__collector}"


class LinkCollector(DefaultCollector):
    """Class that collects information from links and saves to csv.
    This class basically modifies DefaultCollector to be able to group data by link.
    """

    def __init__(self, network_name: str = "default",
                 aggregation_interval: int = 100,
                 custom_path: str | None = None,
                 additional_folders: list[Path] | None = None,
                 params: list[str] | None = None,
                 date: datetime = datetime.now()) -> None:
        if network_name == "default":
            print("Warning: using 'default' as simulation name since the data collector wasn't informed.")
            print("Results will be saved in a default folder and might not be distinguishable from other simulations")

        additional_paths = Path(*additional_folders) if additional_folders is not None else Path()
        path = Path(f"{(custom_path or 'results')}/LinkStepData/{network_name}") / additional_paths

        own_params = list(params or ["Speed"])
        if "TravelTime" in own_params:
            own_params.remove("TravelTime")
            own_params.append("Speed")

        own_params = ["Step", "Link", "Junction", "Junction Type", "Running Vehicles", "Occupancy", "Travel Time"] + own_params
        super().__init__(aggregation_interval, path, own_params)

    def _aggregate(self, curr_value: int) -> None:
        aggregated_df = self._collector_df.groupby(by=self._params[1]).agg('mean').reset_index()
        aggregated_df[self._params[0]] = curr_value # adding step
        aggregated_df[self._params[2]] = self._collector_df[self._params[2]] # adding junction
        aggregated_df[self._params[3]] = self._collector_df[self._params[3]] # adding junction type
        self._update_main_dfs(aggregated_df)

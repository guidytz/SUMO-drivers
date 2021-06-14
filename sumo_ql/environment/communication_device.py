import numpy as np
import random as rd
from datetime import datetime


class CommunicationDevice:
    """Class that is responsible for taking care of the infrastructure information such as the expected rewards for
    the neighboring links.
    """

    def __init__(self, node, max_queue_size, comm_success_rate, environment):
        """Class constructor

        Args:
            node (sumolib.net.node): Node that the CommunicationDevice is installed
            max_queue_size (int): Max number of informed rewards that can be stored in the commDev
            comm_success_rate (float): Value between 0 and 1 that stores the success rate of the communication
            environment (SumoEnvironment): Stores the object of the environment the commDev is in
        """
        self.__node = node
        self.__max_queue_size = max_queue_size
        self.__comm_success_rate = comm_success_rate
        self.__environment = environment
        self.__data = {link.getID(): list() for link in node.getIncoming()}

        rd.seed(datetime.now())

    @property
    def communication_success(self):
        """Property that returns if the communication should fail or succeed based on a random number taken and the 
        success rate stored.

        Returns:
            boolean: Information regarding whether the communication attempt should fail or succeed
        """
        return rd.random() <= self.__comm_success_rate

    def update_stored_rewards(self, link, reward):
        """This method receives a link and a reward, so it stores the reward communicated for the given link. If the 
        queue is already full the oldest reward information is discarded to make room for the newest one to be inserted.
           If the link is not part of the commDev's neighboring links, the method raises a RunTimeError.

        Args:
            link (str): String with link ID to update data
            reward (int): Reward value to insert in link queue
        """
        if link not in self.__data.keys():
            raise RuntimeError("Link is not connected to commDev's node")
        elif len(self.__data[link]) == self.__max_queue_size:
            self.__data[link].pop(0)
        self.__data[link].append(reward)

    def get_expected_reward(self, link):
        """This method returns the expected reward for the given link. If there's no reward stored for the given link, 
        the method returns 0. It returns an average of the stored rewards otherwise.
           If the link is not part of the commDev's neighboring links, the method raises a RunTimeError.

        Args:
            link (str): String with the link ID to get the expected reward

        Returns:
            float: Average of the stored rewards for the given link if there is data stored or 0.0 otherwise
        """
        if link not in self.__data.keys():
            raise RuntimeError("Link is not connected to commDev's node")
        elif len(self.__data[link]) > 0:
            return np.array(self.__data[link]).mean()
        else:
            return 0.0

    def get_outgoing_links_expected_rewards(self):
        """Returns a dictionary containing the expected rewards from all the outgoing links from the commDev's node.

        Returns:
            dict: dictionary containing expected rewards from all outging links from the commDev's node. Being that the 
            keys are the links ID and the values are the expected rewards.
        """
        links_data = dict()
        for link in self.__node.getOutgoing():
            link_id = link.getID()
            neighboring_commDev = self.__environment.get_commDev(
                link.getToNode().getID())
            links_data[link_id] = neighboring_commDev.get_expected_reward(
                link_id)
        return links_data

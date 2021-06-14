import unittest
import unittest.mock as mock
from sumo_ql.environment import CommunicationDevice

class CommunicationDeviceTests(unittest.TestCase):
    
    def setUp(self):
            self.__node = mock.Mock()
            self.__node.getID.return_value = "node1"
            inc_link1 = mock.Mock()
            inc_link1.getID.return_value = "link1"            
            inc_link2 = mock.Mock()
            inc_link2.getID.return_value = "link2"            
            inc_link3 = mock.Mock()
            inc_link3.getID.return_value = "link3"   
            out_link = mock.Mock()
            out_link.getID.return_value = "link1"      
            out_link.getToNode.return_value = self.__node
            self.__node.getIncoming.return_value = list([inc_link1, inc_link2, inc_link3])
            self.__node.getOutgoing.return_value = list([out_link])
            self.__environment = mock.Mock()
            self.__environment.get_commDev.return_value = CommunicationDevice(self.__node, 10, 1, None)
            self.__comm_dev = CommunicationDevice(self.__node, 10, 1, self.__environment)
            

    def test_get_expected_reward(self):
        for _ in range(5):
            self.__comm_dev.update_stored_rewards("link2", -1)
        self.assertEqual(self.__comm_dev.get_expected_reward("link2"), -1)

        for _ in range(5):
            self.__comm_dev.update_stored_rewards("link2", 1)
        self.assertEqual(self.__comm_dev.get_expected_reward("link2"), 0)

        for _ in range(5):
            self.__comm_dev.update_stored_rewards("link2", 1)
        self.assertEqual(self.__comm_dev.get_expected_reward("link2"), 1)

    def test_get_outgoing_links_expected_rewards(self):
        returned_dict = self.__comm_dev.get_outgoing_links_expected_rewards()
        self.__environment.get_commDev.assert_called_with("node1")
        self.assertEqual(returned_dict, {'link1': 0.0})

    def test_communication_success(self):
        comm_dev = CommunicationDevice(self.__node, 10, 1, None)
        self.assertTrue(comm_dev.communication_success)

        comm_dev = CommunicationDevice(self.__node, 10, 0, None)
        self.assertFalse(comm_dev.communication_success)
import unittest
import unittest.mock as mock

from sumo_drivers.environment.sumo_environment import SumoEnvironment
from sumo_drivers.environment.vehicle import Vehicle


class VehicleTest(unittest.TestCase):
    def setUp(self):
        environment = mock.Mock(spec=SumoEnvironment)
        environment.get_link_destination = lambda link: link[-2:]
        environment.is_border_node = lambda node: node == "A3"
        self.__vehicle = Vehicle(vehicle_id="1",
                                 origin="A1",
                                 destination="B5",
                                 arrival_bonus=500,
                                 wrong_destination_penalty=400,
                                 original_route=[],
                                 environment=environment)

    def test_od_pair(self):
        self.assertEqual(self.__vehicle.od_pair, "A1|B5")

    def test_update_current_link(self):
        self.__vehicle.reset()
        self.__vehicle.load_time = 200.0
        with self.assertRaises(RuntimeError):
            self.__vehicle.update_current_link("A1A2", 100.0)

    def test_set_arrival(self):
        with self.assertRaises(RuntimeError):
            self.__vehicle.set_arrival(200.0)

        self.__vehicle.update_current_link("A1A2", 100.0)
        with self.assertRaises(RuntimeError):
            self.__vehicle.set_arrival(50.0)

    def test_route_building(self):
        self.__vehicle.update_current_link("A1A2", 20.0)
        self.assertEqual(self.__vehicle.route, ["A1", "A2"])

        self.__vehicle.update_current_link("A2A3", 60.0)
        self.assertEqual(self.__vehicle.route, ["A1", "A2", "A3"])

        self.__vehicle.set_arrival(80.0)
        self.assertEqual(self.__vehicle.route, ["A1", "A2", "A3"])

    def test_compute_reward(self):
        with self.assertRaises(RuntimeError):
            self.__vehicle.compute_reward()

        self.__vehicle.update_current_link("A1A2", 10.0)
        self.__vehicle.update_current_link("A2A4", 30.0)
        self.assertEqual(self.__vehicle.compute_reward(), -20.0)
        self.__vehicle.set_arrival(200.0)

        self.assertEqual(self.__vehicle.compute_reward(), -420.0)

        self.__vehicle.update_current_link("A4B5", 50.0)
        self.__vehicle.set_arrival(200.0)
        self.assertEqual(self.__vehicle.compute_reward(), 480.0)

    def test_ready_to_act(self):
        self.assertFalse(self.__vehicle.ready_to_act)

        self.__vehicle.update_current_link("A1A2", 10.0)
        self.assertTrue(self.__vehicle.ready_to_act)

        self.__vehicle.update_current_link("A2B2", 10.0)
        self.assertTrue(self.__vehicle.ready_to_act)

        self.__vehicle.update_current_link("B2A3", 10.0)
        self.assertFalse(self.__vehicle.ready_to_act)

        self.__vehicle.update_current_link("A3B5", 10.0)
        self.assertFalse(self.__vehicle.ready_to_act)

    def test_departed(self):
        self.assertFalse(self.__vehicle.departed)

        self.__vehicle.update_current_link("A1A2", 10.0)
        self.assertTrue(self.__vehicle.departed)

    def test_reached_destination(self):
        self.assertFalse(self.__vehicle.reached_destination)

        self.__vehicle.update_current_link("A1A2", 10.0)
        self.__vehicle.set_arrival(200.0)
        self.assertTrue(self.__vehicle.reached_destination)

    def test_travel_time(self):
        with self.assertRaises(RuntimeError):
            _ = self.__vehicle.travel_time

        with self.assertRaises(RuntimeError):
            self.__vehicle.update_current_link("A1A2", 10.0)
            _ = self.__vehicle.travel_time

        self.__vehicle.update_current_link("A1A2", 10.0)
        self.__vehicle.update_current_link("A2A4", 30.0)
        self.__vehicle.update_current_link("A4B5", 80.0)
        self.__vehicle.set_arrival(110.0)
        self.assertEqual(self.__vehicle.travel_time, 100.0)

    def test_is_in_link(self):
        self.__vehicle.update_current_link("A1A2", 10.0)
        self.assertTrue(self.__vehicle.is_in_link("A1A2"))
        self.assertFalse(self.__vehicle.is_in_link("A2B2"))

        self.__vehicle.update_current_link("A2B2", 10.0)
        self.assertTrue(self.__vehicle.is_in_link("A2B2"))


if __name__ == "__main__":
    unittest.main()

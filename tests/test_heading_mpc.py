"""CPU-only regression: use the deployment Python with casadi installed."""
import unittest
import numpy as np
from utils_tasks.tracking_utils import MPC_Controller
from utils_tasks.mode_tracking_utils import ModeMPCController


class HeadingRegression(unittest.TestCase):
    def path(self, degrees):
        t = np.linspace(0, 3, 51)
        a = np.deg2rad(degrees)
        return np.column_stack((t*np.cos(a), t*np.sin(a)))

    def test_reproduce_native_backward_deadlock(self):
        for angle in (-150, -120, 120, 150):
            control, _ = MPC_Controller(self.path(angle)).solve(np.zeros(3))
            self.assertLess(abs(control[1, 0]), 1e-4)
            self.assertLess(abs(control[1, 1]), 1e-4)

    def test_heading_controller_turns_toward_both_sides(self):
        for angle in (-150, -120, 120, 150, 180):
            control, _ = ModeMPCController(self.path(angle)).solve(np.zeros(3))
            self.assertTrue(np.isfinite(control).all())
            self.assertGreater(control[1, 1]*np.sign(angle), .1)
            self.assertGreaterEqual(control[1, 0], -1e-6)
            self.assertLessEqual(abs(control[1, 1]), .500001)

    def test_front_trajectory_still_drives_forward(self):
        control, _ = ModeMPCController(self.path(0)).solve(np.zeros(3))
        self.assertGreater(control[1, 0], .4)
        self.assertLess(abs(control[1, 1]), 1e-4)

    def test_world_rotation_invariance(self):
        yaw = 1.2
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        path = self.path(120) @ rot.T + np.array([10., -5.])
        control, _ = ModeMPCController(path).solve(np.array([10., -5., yaw]))
        self.assertGreater(control[1, 1], .1)


if __name__ == '__main__':
    unittest.main()

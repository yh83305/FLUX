"""MPC tracking with near/far references for explicit-mode NavDP trajectories."""

import casadi as ca
import numpy as np


class ModeMPCController:
    """FLUX MPC dynamics/solver with arc-length lookahead guidance."""

    def __init__(
        self,
        global_planed_traj,
        N=15,
        desired_v=0.5,
        v_max=0.5,
        w_max=0.5,
        ref_gap=3,
        reference_lookahead_distances=(0.0, 0.15, 0.30, 0.55, 0.90),
        far_lookahead_dist=1.2,
        far_position_weight=3.0,
        far_heading_weight=1.0,
    ):
        self.N = int(N)
        self.T = 0.1
        self.desired_v = float(desired_v)
        self.ref_gap = int(ref_gap)
        self.reference_lookahead_distances = np.asarray(
            reference_lookahead_distances, dtype=np.float64
        )
        expected_references = len(range(0, self.N, self.ref_gap))
        if len(self.reference_lookahead_distances) != expected_references:
            raise ValueError(
                "reference_lookahead_distances must match the number of "
                f"tracking stages ({expected_references})"
            )
        if np.any(np.diff(self.reference_lookahead_distances) < 0.0):
            raise ValueError("reference lookahead distances must be nondecreasing")
        self.far_lookahead_dist = float(far_lookahead_dist)
        self.far_position_weight = float(far_position_weight)
        self.far_heading_weight = float(far_heading_weight)
        self.ref_traj = self._prepare_trajectory(global_planed_traj)
        self.ref_traj_len = len(self.reference_lookahead_distances)

        opti = ca.Opti()
        opt_controls = opti.variable(self.N, 2)
        opt_states = opti.variable(self.N + 1, 3)
        v, w = opt_controls[:, 0], opt_controls[:, 1]
        opt_x0 = opti.parameter(3)
        opt_xs = opti.parameter(3 * self.ref_traj_len)
        opt_far_pos = opti.parameter(2)
        opt_far_heading = opti.parameter()

        def dynamics(state, control):
            return ca.vertcat(
                control[0] * ca.cos(state[2]),
                control[0] * ca.sin(state[2]),
                control[1],
            )

        opti.subject_to(opt_states[0, :] == opt_x0.T)
        for i in range(self.N):
            next_state = (
                opt_states[i, :]
                + dynamics(opt_states[i, :], opt_controls[i, :]).T * self.T
            )
            opti.subject_to(opt_states[i + 1, :] == next_state)

        Q = np.diag([10.0, 10.0, 0.0])
        R = np.diag([0.02, 0.15])
        objective = 0
        for i in range(self.N):
            objective += ca.mtimes(
                [opt_controls[i, :], R, opt_controls[i, :].T]
            )
            if i % self.ref_gap == 0:
                reference_index = i // self.ref_gap
                state_error = (
                    opt_states[i, :]
                    - opt_xs[reference_index * 3:reference_index * 3 + 3].T
                )
                objective += ca.mtimes([state_error, Q, state_error.T])

        far_error = opt_states[-1, 0:2] - opt_far_pos.T
        objective += self.far_position_weight * ca.sumsqr(far_error)
        heading_error = ca.atan2(
            ca.sin(opt_states[-1, 2] - opt_far_heading),
            ca.cos(opt_states[-1, 2] - opt_far_heading),
        )
        objective += self.far_heading_weight * heading_error ** 2
        opti.minimize(objective)

        opti.subject_to(opti.bounded(0.0, v, v_max))
        opti.subject_to(opti.bounded(-w_max, w, w_max))
        opti.solver("ipopt", {
            "ipopt.max_iter": 100,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.acceptable_tol": 1e-8,
            "ipopt.acceptable_obj_change_tol": 1e-6,
        })

        self.opti = opti
        self.opt_xs = opt_xs
        self.opt_x0 = opt_x0
        self.opt_far_pos = opt_far_pos
        self.opt_far_heading = opt_far_heading
        self.opt_controls = opt_controls
        self.opt_states = opt_states
        self.last_opt_x_states = None
        self.last_opt_u_controls = None

    @staticmethod
    def _prepare_trajectory(trajectory):
        points = np.asarray(trajectory, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 2 or len(points) == 0:
            raise ValueError("global planned trajectory must be [H,2+] and nonempty")
        points = points[:, :2]
        if len(points) == 1:
            points = np.repeat(points, 2, axis=0)
        return points

    @staticmethod
    def _arc_lengths(points):
        return np.concatenate((
            np.zeros(1, dtype=np.float64),
            np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)),
        ))

    @staticmethod
    def _sample_arc(points, arc_lengths, distances):
        query = np.clip(np.asarray(distances), arc_lengths[0], arc_lengths[-1])
        return np.stack((
            np.interp(query, arc_lengths, points[:, 0]),
            np.interp(query, arc_lengths, points[:, 1]),
        ), axis=1)

    def find_reference_traj(self, x0, global_planed_traj=None):
        points = self.ref_traj if global_planed_traj is None else self._prepare_trajectory(
            global_planed_traj
        )
        arc = self._arc_lengths(points)
        nearest = int(np.argmin(np.linalg.norm(points - np.asarray(x0[:2]), axis=1)))
        start_arc = arc[nearest]
        references = self._sample_arc(
            points, arc, start_arc + self.reference_lookahead_distances
        )
        far_arc = min(start_arc + self.far_lookahead_dist, arc[-1])
        far_pos = self._sample_arc(points, arc, [far_arc])[0]
        tangent_radius = 0.05
        tangent_points = self._sample_arc(
            points,
            arc,
            [max(start_arc, far_arc - tangent_radius),
             min(arc[-1], far_arc + tangent_radius)],
        )
        tangent = tangent_points[1] - tangent_points[0]
        if np.linalg.norm(tangent) < 1e-8:
            fallback_start = max(0, nearest - 1)
            fallback_end = min(len(points) - 1, nearest + 1)
            tangent = points[fallback_end] - points[fallback_start]
        far_heading = float(np.arctan2(tangent[1], tangent[0]))
        return references, far_pos, far_heading

    def solve(self, x00):
        references, far_pos, far_heading = self.find_reference_traj(x00)
        reference_states = np.concatenate(
            (references, np.zeros((len(references), 1))), axis=1
        ).reshape(-1, 1)
        self.opti.set_value(self.opt_xs, reference_states)
        self.opti.set_value(self.opt_x0, x00)
        self.opti.set_value(self.opt_far_pos, far_pos)
        self.opti.set_value(self.opt_far_heading, far_heading)

        controls = (
            np.zeros((self.N, 2))
            if self.last_opt_u_controls is None else self.last_opt_u_controls
        )
        states = (
            np.zeros((self.N + 1, 3))
            if self.last_opt_x_states is None else self.last_opt_x_states
        )
        self.opti.set_initial(self.opt_controls, controls)
        self.opti.set_initial(self.opt_states, states)
        solution = self.opti.solve()
        self.last_opt_u_controls = solution.value(self.opt_controls)
        self.last_opt_x_states = solution.value(self.opt_states)
        return self.last_opt_u_controls, self.last_opt_x_states

    def reset(self):
        self.last_opt_x_states = None
        self.last_opt_u_controls = None

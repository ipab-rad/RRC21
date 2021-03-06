#!/usr/bin/env python3
from mp.utils import repeat
from mp.align_rotation import get_yaw_diff
from mp.const import TRANSLU_CYAN, CUBOID_SIZE, INIT_JOINT_CONF
from scipy.spatial.transform import Rotation as R
import numpy as np


def complete_keypoints(start, goal, unit_length=0.008):
    """complete_keypoints.
    Fills in intermediate keypoints between start and goal, each separated by
    `unit_length`

    Args:
        start: start tip position
        goal: goal tip position
        unit_length: unit length of separation between tip positions
    """
    assert start.shape == goal.shape
    assert len(start.shape) in [1, 2]
    diff = goal - start
    if len(start.shape) == 2:
        length = max(np.linalg.norm(diff, axis=1))
    else:
        length = np.linalg.norm(diff)

    num_keypoints = int(length / unit_length)
    keypoints = [start + diff * i / num_keypoints for i in range(num_keypoints)]
    return keypoints


class ScriptedActions(object):
    """
    ScriptedActions.
    This class implements certain scripted action sequences that are commonly
    utuilised in the movement of the trifinger robot fingers.

    Namely:
        `add_heuristic_pregrasp`: moves the fingers to a given
        pre-heuristic-grasp position
        `add_grasp`: moves the fingers to a given grasp position
        `add_move`: moves the fingers in a certain way - this is a primitive
        funtion used by all other movement primitives
        `add_release`: releases grasp
        `add_move_to_center`
        `add_raise_tips`

    """

    def __init__(self, env, robot_tip_positions, grasp, vis_markers=None):
        self.env = env
        self.grasp = grasp
        self.robot_tip_positions = robot_tip_positions
        self.tip_positions_list = []
        self.executed = False
        self.vis_markers = vis_markers
        self._markers = set()

    def _update_markers(self, target_tip_positions, marker_name,
                        color=TRANSLU_CYAN):
        if self.vis_markers is not None:
            if marker_name in self._markers:
                self.vis_markers.remove()
            self.vis_markers.add(target_tip_positions, color=color)
            self._markers.add(marker_name)

    def add_move(self, tip_pos, unit_length, min_height=0.01):
        """add_move.
        Adds tipnposition waypoints to the `tip_positions_list`. It calls the
        `complete_keypoints` to create intermediate waypoints between start and
        goal positions

        Args:
            tip_pos: goal tip positions
            unit_length: unit length of movement between each keypoint
            min_height: minimum height the robot tips can reach
        """
        current_tip_pos = self.get_last_tippos()

        if np.any(tip_pos[:, 2] < min_height):
            tip_pos[:, 2] = np.maximum(tip_pos[:, 2], min_height)

        # do not move tip_pos if the corresponding tip is invalid
        mask = np.eye(3)[self.grasp.valid_tips, :].sum(0).reshape(3, -1)
        tip_pos = tip_pos * mask + (1 - mask) * self.robot_tip_positions
        self.tip_positions_list += complete_keypoints(current_tip_pos, tip_pos,
                                                      unit_length=unit_length)

    def add_grasp(self, coef=0.9):
        target_tip_positions = self.grasp.T_cube_to_base(
            self.grasp.cube_tip_pos * coef
        )
        self._update_markers(target_tip_positions, 'grasp')
        self.add_move(target_tip_positions, 0.004)

    def add_release(self, coef=2.0, min_height=0.01):
        target_tip_positions = self.grasp.T_cube_to_base(
            self.grasp.cube_tip_pos * coef
        )
        if np.any(target_tip_positions[:, 2] < min_height):
            target_tip_positions[:, 2] = np.maximum(target_tip_positions[:, 2], min_height)

        self.add_move(target_tip_positions, 0.004)

    def add_release2(self, coef=2.0, min_height=0.01):
        """
        'add_release' method requires 'grasp.cube_tip_pos' to calculate target tip positions.
        However this one calculates the target positions only from current tip positions.
        """
        tip_pos = self.get_last_tippos()
        center = np.mean(tip_pos, axis=0)
        target_tip_positions = center + (tip_pos - center) * coef
        if np.any(target_tip_positions[:, 2] < min_height):
            target_tip_positions[:, 2] = np.maximum(target_tip_positions[:, 2], min_height)
        self.add_move(target_tip_positions, 0.004)

    def add_move_to_center(self, coef=0.6):
        self.grasp.update(np.zeros(3), self.grasp.quat)
        self.add_move(
            self.grasp.T_cube_to_base(self.grasp.cube_tip_pos * coef),
            0.004
        )

    def add_raise_tips(self, height=CUBOID_SIZE[0] * 1.5):
        """
        scripted action sequence to raise the tips of the robiot fingers
        """
        # get last tip positions
        target_tip_pos = self.get_last_tippos()

        # increase the height of the tip postions by a certain height
        target_tip_pos[:, 2] = height

        # generate actions to raise the height of the robot fingers
        self.add_move(target_tip_pos, 0.004)

    def add_heuristic_pregrasp(self, pregrasp_tip_pos):
        if self.get_last_tippos()[:, 2].min() < CUBOID_SIZE[0] / 2:
            print('Warning: adding heuristic pregrasp even though robot_tip postiion is low')
        above_target_tip_positions = np.copy(pregrasp_tip_pos)
        above_target_tip_positions[:, 2] = CUBOID_SIZE[0] * 1.5
        self.add_move(above_target_tip_positions, 0.004)
        self.add_move(pregrasp_tip_pos, 0.004)

    def add_pitch_rotation(self, height, rotate_axis, rotate_angle, coef=0.6):
        # lift cube up
        self.grasp.update(self.grasp.pos + np.array([0, 0, height]),
                          self.grasp.quat)
        target_tip_positions = self.grasp.T_cube_to_base(
            self.grasp.cube_tip_pos * coef
        )
        self._update_markers(target_tip_positions, 'liftup')
        self.add_move(target_tip_positions, 0.004)

        # rotate cube
        rotate_step = np.sign(rotate_angle) * np.pi / 30
        rot = R.from_rotvec(rotate_axis * rotate_step)
        print(f'add_pitch_rotation: rotate_axis {rotate_axis}\trotate_angle {rotate_angle}')
        for _ in range(int(rotate_angle / rotate_step)):
            orientation = (R.from_quat(self.grasp.quat) * rot).as_quat()
            self.grasp.update(self.grasp.pos, orientation)
            target_tip_positions = self.grasp.T_cube_to_base(
                self.grasp.cube_tip_pos * coef
            )
            self.tip_positions_list.append(target_tip_positions)

        # place_cube
        self.grasp.update(self.grasp.pos - np.array([0, 0, height]),
                          self.grasp.quat)
        self.add_move(
            self.grasp.T_cube_to_base(self.grasp.cube_tip_pos * coef),
            0.004
        )

    def add_yaw_rotation(self, goal_quat, step_angle=np.pi/3, coef=0.9):
        angle = get_yaw_diff(self.grasp.quat, goal_quat)
        angle_clipped = np.clip(angle, -step_angle, step_angle)
        ori = (
            R.from_euler('Z', angle_clipped)
            * R.from_quat(self.grasp.quat)
        ).as_quat()
        self.grasp.update(self.grasp.pos, ori)
        target_tip_positions = self.grasp.T_cube_to_base(
            self.grasp.cube_tip_pos * coef
        )
        self._update_markers(target_tip_positions, 'yaw')
        self.add_move(target_tip_positions, 0.002)
        return angle_clipped

    def get_last_tippos(self):
        if len(self.tip_positions_list) == 0:
            return np.copy(self.robot_tip_positions)
        else:
            return np.copy(self.tip_positions_list[-1])

    def _tip_positions_to_actions(self):
        """_tip_positions_to_actions.
        Applies inverse kinematics and generates joint positions from the
        given tip positions. Tip positions are stored in
        `self.tip_positions_list` list. This list is updated by the `add_move`
        function
        """
        ik = self.env.pinocchio_utils.inverse_kinematics

        actions = []
        skip_count = 0
        for tip_positions in self.tip_positions_list:
            q = INIT_JOINT_CONF.copy()
            for i in range(3):
                q = ik(i, tip_positions[i], q)
                if q is None:
                    print('Warning: IK solution not found (tip_positions_to_actions)')
                    break
            if q is not None:
                for _ in range(skip_count + 1):
                    actions.append(q)
                skip_count = 1
            else:
                skip_count += 1
        return actions

    def get_action_sequence(self, frameskip=1, action_repeat=1, action_repeat_end=1):
        """get_action_sequence.
        Generates action sequence for the robot joints to follow along.
        Converts the tip positions to robot joint positions, adds them to a
        list and then returns this list of joint and tip positions

        Args:
            frameskip:
            action_repeat:
            action_repeat_end:
        """
        action_seq = self._tip_positions_to_actions()
        action_seq = repeat(action_seq, action_repeat)
        action_seq += repeat([action_seq[-1]], action_repeat_end)

        action_seq = repeat(action_seq, frameskip)
        return action_seq

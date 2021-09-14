"""
File: move_finger.py
Author: Aditya Kamireddypalli
Email: a.kamireddypalli@sms.ed.ac.uk
Github: https://github.com/kzernobog
Description:
    This file tries to implement a state machine to move a single
    finger of the `trifinger` robot. Amongst other things it tries to stick to the
    API structure opf the overall state/statemachine architecture.
"""

from mp.action_sequences import ScriptedActions
from mp import states
from .states import State, StateMachine
import numpy as np
from .fingers import get_finger_configuration, get_finger_configuration_dice
from mp import grasping
from mp.const import CONTRACTED_JOINT_CONF, INIT_JOINT_CONF
from dice_pose.estimate_dice_pose import DicePose

POS1 = np.array([0.0, 1.4, -2.4, 0.0, 1.4, -2.4, 0.0, 1.4, -2.4], dtype=np.float32)
POS2 = np.array([0.0, 1.4, -2.2, 0.0, 1.4, -2.4, 0.0, 1.4, -2.4], dtype=np.float32)
POS3 = np.array([0.0, 1.4, -1.9, 0.0, 1.4, -2.4, 0.0, 1.4, -2.4], dtype=np.float32)

############
#  States  #
############


class OpenLoopState(State):
    """Base class for open-loop control states."""
    def __init__(self, env):
        self.env = env
        self.next_state = None
        self.failure_state = None
        self.reset()

    def reset(self):
        self.actions = None

    def connect(self, next_state, failure_state):
        self.next_state = next_state
        self.failure_state = failure_state

    def get_action_generator(self, obs, info):
        """Yields (action, info) tuples."""
        raise NotImplementedError

    def __call__(self, obs, info=None):
        info = dict() if info is None else info
        try:
            if self.actions is None:
                self.actions = self.get_action_generator(obs, info)
            action, info = self.actions.__next__()
            return action, self, info
        except Exception as e:
            self.reset()
            if isinstance(e, StopIteration):
                # query the next state
                return self.next_state(obs, info)
            else:
                print(f"Caught error: {e}")
                return self.get_action(frameskip=0), self.failure_state, info

class MoveFingerState(OpenLoopState):

    """Docstring for MoveFinger. """
    def __init__(self, env, steps=300):
        super().__init__(env)
        self.steps = steps

    def get_action_generator(self, obs, info):
        """
        Generates actions for the `move_finger` state within the context of
        the state machine.

        """
        # Get a trajectory for one finger to the center of a face
        # You can look at how the grasps are generated within
        # `HeuristicGraspState` and emulate the trajectory generation
        # Execute the trajectory until it touches the face. You can copy how
        # approach actions are generated in `HeuristicGraspState` and use the
        # same functionality

        # retrieve object position
        for pos in [POS1, POS2, POS3]:
            yield self.get_action(position=pos, frameskip=self.steps // 2), info

class MoveFingerToObjState(OpenLoopState):

    """Docstring for MoveFingerToObjState. """
    def __init__(self, env, steps=300):
        super().__init__(env)
        self.steps = steps

    def get_action_generator(self, obs, info):
        """
        Generates actions for the `move_finger_obj` state within the context of
        the state machine. This state is responsible for moving the finger to
        the object

        """
        # Get a trajectory for one finger to the center of a face
        # You can look at how the grasps are generated within
        # `HeuristicGraspState` and emulate the trajectory generation
        # Execute the trajectory until it touches the face. You can copy how
        # approach actions are generated in `HeuristicGraspState` and use the
        # same functionality

        # retrieve object position
        # for pos in [POS1, POS2, POS3]:
        #     yield self.get_action(position=pos, frameskip=self.steps // 2), info

        # TODO: need to ensure that fing_mov is a `Grasp` object
        __import__('pudb').set_trace()
        fing_mov = get_finger_configuration(self.env,
                                           obs['object_position'],
                                           obs['object_orientation'])

        # sticking with the original solution's `Trifinger` API
        info['grasp'] = fing_mov[0]
        actions = grasping.get_grasp_approach_actions(self.env, obs,
                                                      fing_mov[0],
                                                      move_finger=True)
        for pos in actions:
            yield self.get_action(position=pos, frameskip=1), info

class MoveFingerForDice(OpenLoopState):
    def __init__(self, env, steps=300):
        super().__init__(env)
        self.steps = steps
        self.pose = None

    def get_action_generator(self, obs, info):
        if self.pose is None:
            self.pose = self.env.yield_pose()
        try:
            pose = self.pose.__next__()
        except Exception as e:
            self.reset()
            self.pose = None
            if isinstance(e, StopIteration):
                return self.next_state(obs, info)
            else:
                print(f"Caught error: {e}")
                return self.get_action(frameskip=0), self.failure_state, info

        __import__('pudb').set_trace()
        fing_mov = get_finger_configuration_dice(self.env, pose, (0, 0, 0, 1))
        info['grasp'] = fing_mov[0]
        actions = grasping.get_grasp_approach_actions(self.env, obs,
                                                     fing_mov[0],
                                                     move_finger=True)
        for pos in actions:
            yield self.get_action(position=pos, frameskip=1), info


class EstimateDicePose(OpenLoopState):
    def __init__(self, env, steps=300):
        super().__init__(env)
        self.steps = steps

    def get_action_generator(self, obs, info):
        self._estimate_pose(obs)
        # __import__('pudb').set_trace()
        for pos in [POS1, POS2, POS3]:
            yield self.get_action(position=pos, frameskip=self.steps // 2), info

    def _estimate_pose(self, obs):
        pose_estimator = DicePose(obs['camera_images'], obs['achieved_goal'])
        pose_estimator.estimate()
        dice_pose = pose_estimator.resolve()
        self.env.set_pose_queue(dice_pose)
        return






####################
#  State Machines  #
####################

class PositionControlStateMachine(StateMachine):
    def build(self):
        """
        Builds the experimental state machine
        """
        self.goto_init_state = states.GoToInitPoseState(self.env)
        # self.move_finger = MoveFingerState(self.env)
        self.move_finger_obj = MoveFingerToObjState(self.env)
        self.wait = states.WaitState(self.env, 300)
        self.failure = states.FailureState(self.env)

        # define state trasitions
        self.goto_init_state.connect(next_state=self.move_finger_obj, failure_state=self.failure)
        self.move_finger_obj.connect(next_state=self.wait, failure_state=self.failure)
        self.wait.connect(next_state=self.goto_init_state,
                          failure_state=self.failure)
        return self.goto_init_state

class DicePoseEstimationStateMachine(StateMachine):
    def build(self):
        """
        builds state machine to move fingers to intialstate. This is mainly to
        experiment with better dice pose estimation.
        """

        self.goto_init_state = states.GoToInitPoseState(self.env)
        self.movefinger = MoveFingerForDice(self.env)
        self.estimatepose = EstimateDicePose(self.env)
        self.wait = states.WaitState(self.env, 300)
        self.failure = states.FailureState(self.env)

        # connect these state up
        self.goto_init_state.connect(next_state=self.estimatepose,
                                     failure_state=self.failure)
        self.estimatepose.connect(next_state=self.movefinger,
                                  failure_state=self.failure)
        self.movefinger.connect(next_state=self.wait,
                                failure_state=self.failure)
        self.wait.connect(next_state=self.estimatepose,
                          failure_state=self.failure)
        return self.goto_init_state

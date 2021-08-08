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
        object_pos = obs[""]


####################
#  State Machines  #
####################

class PositionControlStateMachine(StateMachine):
    def build(self):
        """
        Builds the experimental state machine
        """
        self.goto_init_state = states.GoToInitPoseState(self.env)
        self.wait = states.WaitState(self.env, 30)
        self.failure = states.FailureState(self.env)

        # define state trasitions
        self.goto_init_state.connect(next_state=self.wait, failure_state=self.failure)
        self.wait.connect(next_state=self.goto_init_state,
                          failure_state=self.failure)
        return self.goto_init_state

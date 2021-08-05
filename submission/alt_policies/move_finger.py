"""
File: move_finger.py
Author: yourname
Email: yourname@email.com
Github: https://github.com/yourname
Description: This file tries to implement a state machine to move a single
finger of the `trifinger` robot. Amongst other things it tries to stick to the
API structure opf the overall state/statemachine architecture.
"""

from mp.action_sequences import ScriptedActions
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


####################
#  State Machines  #
####################

class PositionControlStateMachine(StateMachine):
    def build(self):
        self.init_state = MoveFingerState()
        return self.init_state

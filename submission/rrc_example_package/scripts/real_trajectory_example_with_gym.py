#!/usr/bin/env python3
"""Demo on how to run the robot using the Gym environment

This demo creates a RealRobotCubeTrajectoryEnv environment and runs one episode
using a dummy policy.
"""
import json
import sys
from mp.utils import set_seed

from trifinger_simulation.tasks import move_cube_on_trajectory as task
from combined_code import create_state_machine
from env.make_env import make_env
from rrc_example_package import cube_trajectory_env
from rrc_example_package.example import PointAtTrajectoryPolicy
# goal = {
#     "_goal":  [
#         [0, [0, 0, 0.08]],
#         [10000, [0, 0.07, 0.08]],
#         [20000, [0.07, 0.07, 0.08]],
#         [30000, [0.07, 0, 0.08]],
#         [40000, [0.07, -0.07, 0.08]],
#         [50000, [0, -0.07, 0.08]],
#         [60000, [-0.07, -0.07, 0.06]],
#         [70000, [-0.07, 0, 0.08]],
#         [80000, [-0.07, 0.07, 0.08]],
#         [90000, [0, 0.07, 0.08]],
#         [100000, [0, 0, 0.08]]
#     ]
# }


def _init_env(goal_pose_dict, difficulty):
    eval_config = {
        'action_space': 'torque_and_position',
        'frameskip': 3,
        'reward_fn': 'compute_reward',
        'termination_fn': 'no_termination',
        'initializer': 'random_init',
        'monitor': False,
        'episode_length': task.EPISODE_LENGTH,
        'visualization': False,
        'sim': False,
        'rank': 0
    }

    set_seed(0)
    env = make_env(goal_pose_dict, difficulty, **eval_config)
    return env

class RandomPolicy:
    """Dummy policy which uses random actions."""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation):
        return self.action_space.sample()


def main():
    # the goal is passed as JSON string
    print("Booyahkasha")
    goal_json = sys.argv[1]
    print("json file: {}".format(goal_json))
    goal = json.loads(goal_json)
    print('goal: {}'.format(goal))

    # env = cube_trajectory_env.RealRobotCubeTrajectoryEnv(
    #     goal_difficulty=3,
    #     goal_trajectory=goal,
    #     action_type=cube_trajectory_env.ActionType.TORQUE_AND_POSITION,
    #     step_size=1,
    #     simulation=True,
    # )
    env = _init_env(goal['_goal'], 3)

    # policy = RandomPolicy(env.action_space)
    # policy = PointAtTrajectoryPolicy(env.action_space, goal)
    state_machine = create_state_machine(3, 'mp-pg', env, False, False)

    observation = env.reset()
    state_machine.reset()
    t = 0
    is_done = False
    accumulated_reward = 0
    while not is_done:
        action = state_machine(observation)
        observation, reward, is_done, info = env.step(action)
        t = info["time_index"]
        print("reward:", reward)

        accumulated_reward += reward


if __name__ == "__main__":
    main()

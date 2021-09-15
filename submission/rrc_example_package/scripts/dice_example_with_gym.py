#!/usr/bin/env python3
"""Demo on how to run the robot using the Gym environment

This demo creates a RealRobotRearrangeDiceEnv environment and runs one episode
using a dummy policy.
"""
import json
import sys

import env.wrappers as wrappers
from combined_code import create_machine
from rrc_example_package import rearrange_dice_env
from rrc_example_package.example import PointAtDieGoalPositionsPolicy
import trifinger_simulation
from trifinger_simulation.tasks import move_cube_on_trajectory as task


def make_env(goal):
    # eval_config = {
    #     'action_space': 'torque_and_position',
    #     'frameskip': 3,
    #     'reward_fn': 'compute_reward',
    #     'termination_fn': 'no_termination',
    #     'initializer': 'random_init',
    #     'monitor': False,
    #     'episode_length': task.EPISODE_LENGTH,
    #     'visualization': False,
    #     'sim': False,
    #     'rank': 0
    # }
    eval_config = {
        'frameskip': 3,
        'episode_length': task.EPISODE_LENGTH,
        'visualization': False,
        'sim': False,
        'rank': 0
    }
    env = rearrange_dice_env.RealRobotRearrangeDiceEnv(
        goal=goal,
        action_type=rearrange_dice_env.ActionType.TORQUE_AND_POSITION,
    )
    env = wrappers.AdaptiveActionSpaceWrapper(env)
    return env

def main():
    # the goal is passed as JSON string
    goal_json = sys.argv[1]
    goal = json.loads(goal_json)

    # goal = trifinger_simulation.tasks.rearrange_dice.sample_goal()

    # env = rearrange_dice_env.RealRobotRearrangeDiceEnv(
    #     goal,
    #     rearrange_dice_env.ActionType.POSITION,
    #     step_size=1,
    # )
    #################################################
    #  uncomment the following when using real bot  #
    #################################################
    # env = rearrange_dice_env.RealRobotRearrangeDiceEnv(
    #     goal,
    #     rearrange_dice_env.ActionType.POSITION,
    #     step_size=1,
    #     sim=False,
    #     vis=False
    # )

    # policy = PointAtDieGoalPositionsPolicy(env.action_space, goal)
    env = make_env(goal)

    machine = create_machine(env)

    observation = env.reset()
    machine.reset()

    t = 0
    is_done = False
    while not is_done:
        # action = policy.predict(observation, t)
        action = machine(observation)
        observation, reward, is_done, info = env.step(action)
        t = info["time_index"]


if __name__ == "__main__":
    main()

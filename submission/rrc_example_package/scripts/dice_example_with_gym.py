#!/usr/bin/env python3
"""Demo on how to run the robot using the Gym environment

This demo creates a RealRobotRearrangeDiceEnv environment and runs one episode
using a dummy policy.
"""
import json
import sys

from combined_code import create_machine
from rrc_example_package import rearrange_dice_env
from rrc_example_package.example import PointAtDieGoalPositionsPolicy
import trifinger_simulation


def main():
    # the goal is passed as JSON string
    goal_json = sys.argv[1]
    goal = json.loads(goal_json)

    # goal = trifinger_simulation.tasks.rearrange_dice.sample_goal()

    env = rearrange_dice_env.RealRobotRearrangeDiceEnv(
        goal,
        rearrange_dice_env.ActionType.POSITION,
        step_size=1,
    )
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

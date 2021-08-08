"""
This script is an example script, that plays around with different trifinger
control policies. It holds boilerplate code to ezxperiment with different state
machines and environment policies.
"""
import sys

from rrc_example_package import cube_trajectory_env as cb

def main():
    env = cb.SimCubeTrajectoryEnv(3)
    state_machine = create_machine(env)

    observation = env.reset()
    state_machine.reset()
    is_done = False
    accumulated_reward = 0
    t = 0
    while not is_done:
        action = state_machine(observation)
        observation, reward, is_done = env.step(action)
        accumulated_reward += reward


if __name__ == "__main__":
    main()

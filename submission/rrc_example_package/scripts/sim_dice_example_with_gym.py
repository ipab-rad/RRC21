"""Demo on how to run the simulation using the Gym environment
This demo creates a SimRearrangeDiceEnv environment and runs one episode using
a dummy policy.
"""
import env.wrappers as wrappers
from rrc_example_package import rearrange_dice_env
from rrc_example_package.example import PointAtDieGoalPositionsPolicy
from combined_code import create_machine
from trifinger_simulation.tasks import move_cube_on_trajectory as task

def make_env():
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
        'visualization': True,
        'sim': True,
        'rank': 0
    }
    env = rearrange_dice_env.SimRearrangeDiceEnv(
        goal=None,
        action_type=rearrange_dice_env.ActionType.TORQUE_AND_POSITION,
    )
    env = wrappers.AdaptiveActionSpaceWrapper(env)
    return env


def main():
    # env = rearrange_dice_env.SimRearrangeDiceEnv(
    #     goal=None,  # passing None to sample a random goal
    #     action_type=rearrange_dice_env.ActionType.TORQUE_AND_POSITION,
    #     visualization=True,
    # )
    env = make_env()

    machine = create_machine(env)
    is_done = False
    observation = env.reset()
    machine.reset()
    t = 0

    # policy = PointAtDieGoalPositionsPolicy(env.action_space, env.current_goal)
    # __import__('pudb').set_trace()
    while not is_done:
        # action = policy.predict(observation, t)
        action = machine(observation)
        # __import__('pudb').set_trace()
        observation, reward, is_done, info = env.step(action)
        t = info["time_index"]


if __name__ == "__main__":
    main()

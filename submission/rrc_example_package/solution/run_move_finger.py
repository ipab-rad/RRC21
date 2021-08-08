
#!/usr/bin/env python3
"""Run a single episode with a controller in simulation."""
import argparse
import sys
import json

from env.make_env import make_env
from trifinger_simulation.tasks import move_cube_on_trajectory as task
from mp.utils import set_seed
from combined_code import create_state_machine

goal_dict = {
    "_goal":  [
        [0, [0, 0, 0.08]],
        [10000, [0, 0.07, 0.08]],
        [20000, [0.07, 0.07, 0.08]],
        [30000, [0.07, 0, 0.08]],
        [40000, [0.07, -0.07, 0.08]],
        [50000, [0, -0.07, 0.08]],
        [60000, [-0.07, -0.07, 0.06]],
        [70000, [-0.07, 0, 0.08]],
        [80000, [-0.07, 0.07, 0.08]],
        [90000, [0, 0.07, 0.08]],
        [100000, [0, 0, 0.08]]
    ]
}

def _init_env(goal_pose_dict, difficulty):
    eval_config = {
        'action_space': 'torque_and_position',
        'frameskip': 3,
        'reward_fn': 'compute_reward',
        'termination_fn': 'no_termination',
        'initializer': 'random_init',
        'monitor': False,
        'visualization': True,
        'sim': True,
        'rank': 0
    }

    set_seed(0)
    env = make_env(goal_pose_dict, difficulty, **eval_config)
    return env


def main():
    parser = argparse.ArgumentParser('args')
    parser.add_argument('difficulty', help="specify difficulty level of task \
                        (valid only for 2020 version of the competition)", type=int, default=3)

    # parser.add_argument(
    #     "trajectory",
    #     type=json.loads,
    #     metavar="JSON",
    #     help="Goal trajectory as a JSON string.",
    # )
    parser.add_argument('method', type=str, help="The method to run. One of 'mp-pg', 'cic-cg', 'cpc-tg'")
    parser.add_argument('--residual', default=False, action='store_true',
                        help="add to use residual policies. Only compatible with difficulties 3 and 4.")
    parser.add_argument('--bo', default=False, action='store_true',
                        help="add to use BO optimized parameters.")
    args = parser.parse_args()
    # print('goal: {}'.format(args.trajectory))
    # sys.exit()
    # goal_pose = task.sample_goal()
    goal_pose = goal_dict['_goal']
    # print('goal: {}'.format(goal_pose))
    # sys.exit()
    # goal_pose_dict = {
    #     'position': goal_pose.position.tolist(),
    #     'orientation': goal_pose.orientation.tolist()
    # }

    # goal_pose_trajectory = goal_dict['_goal']

    # env = _init_env(goal_pose_trajectory, args.difficulty)
    env = _init_env(goal_pose, 3)
    state_machine = create_state_machine(3, args.method, env,
                                         args.residual, args.bo)

    #####################
    # Run state machine
    #####################
    obs = env.reset()
    state_machine.reset()

    done = False
    while not done:
        action = state_machine(obs)
        obs, _, done, _ = env.step(action)


if __name__ == "__main__":
    main()

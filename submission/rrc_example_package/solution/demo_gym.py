from rrc_example_package import cube_trajectory_env
from rrc_example_package.solution.utils.policy import RandomPolicy


def main():
    env = cube_trajectory_env.SimCubeTrajectoryEnv(
        goal_trajectory=None, 
        action_type=cube_trajectory_env.ActionType.POSITION, 
        step_size=100,
        visualization=True
    )

    policy = RandomPolicy(env.action_space)
    observation = env.reset()
    is_done = False
    while not is_done:
        action = policy.predict(observation)
        observation, reward, is_done, info = env.step(action)
        print("reward: {}".format(reward))

if __name__ == "__main__":
   main()

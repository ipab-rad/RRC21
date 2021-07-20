# Task inbox
- [ ] Compute proper reward
- [ ] ensure episode length is proper
  - [ ] what is the deal with steps
- [ ] Print the final reward
# Tasks
- [X] input goal trajectory
  - [X] ensure that the script reads the proper goal trajectory
- [X] What is the deal with time steps
- [X] How are goals switched in the trajectory in the original script
  - [X] `env` has a goal variable that stores the current goal - create a new
    variable in new `env` to store trajectory. `self.info` can do this
  - [X] The `create_observation` function updates the observation state to have
    current goal - every time this function is called, you can update the
    `self.goal` value
    - [X] Ensure that the `self.goal` is a dictionary with `position` and
      `orientation` attributes
  - [X] maintain a `self.info` dictionary which stores the time index and
    trajectory - initialise in `reset`
- [X] Use the example `env` and modify how the reward is being computed
- [X] How is the state machine informed about these goals
- [X] Running submission checklist
  - [X] turn visualisation false
- [ ] pull updated benchmark repo into another branch
  - [ ] integrate and test `cic-cg` and `cpc-tg` from original repo
- [ ] submit existing running solution to real robot.
  - [ ] Check what action type is being sent
    - [ ] what is the action type to be sent to the robot
  - [ ] Check if backend starts sending observations according to the felix's
    reply

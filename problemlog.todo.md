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
- [X] pull updated benchmark repo into another branch
  - [X] integrate and test `cic-cg` and `cpc-tg` from original repo
- [X] submit existing running solution to real robot.
  - [X] Check what action type is being sent
    - [X] what is the action type to be sent to the robot
  - [X] Check if backend starts sending observations according to the felix's
    reply
- [X] streamline data and logs from experiments
  - [X] download and process data
    - [X] create necessary csvs
    - [X] create necessary video files for visualisation
- [ ] Experiment with different coeff for `add_grasp` functionlity in
  `grasp_motion`
- [ ] Sort out th code Cartesian Position control
    - [ ] no action going through in `GoToInitState` - pass position into
      action
- [ ] Sort out the code Cartesian Impedence control
- [X] Move one finger to a tip position on the cube
  - [X] select which of the three estimated tip position you want to move
    towards.
    - [X] house this in `MoveFingerToObjState` 
  - [X] select which finger needs to go there
  - [X] Create a Grasp object of the finger tips
  - [X] generate approach actions
  - [X] Execute actions
  - [X] Execute touch not push
- [ ] Construct pose for dice 2021-08-26 15:00
  - [X] extract segmentation masks 2021-08-26 15:01
  - [X] visualise segmentation mask overlay over the image
    - [X] include segmentation mask as an alpha channel for the image - does not display it
    - [X] OpenCV does not imshow alpha channel, copied a wrapper from stackoverflow
  - [ ] retrieve numbers on dice
    - [ ] retrieve segmentation blobs
    - [ ] extract grayscale intensity values from the blobs
    - [ ] Threshold to demarcate numbers.
    - [ ] Extract number points
  - [ ] extract dice in each camera
    - [X] connected components on the segmentation mask
    - [ ] separate dice that are shown as one component - idea is just get dice with black background
  - [ ] associate dice in each camera
  - [X] project imaginary dice on the image
    - [X] project cube onto image
    - [X] use connected component blobs to isolate each single dice
    - [X] project imaginary 3d dice on image
    - [X] choose points on die and on blob
    - [X] calculate hausdorff distance.
    - [X] optimize distance
      - [X] code up the optmisation loop
  - [ ] resolve pose estimation for segmentation blobs with multiple dice
  - [ ] ensure that `ProjectCube` takes in correct camera parameters
- [ ] What is the reason for frameskip
- [ ] make state machine consumable by dice env

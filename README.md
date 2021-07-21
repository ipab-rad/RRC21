# Simulation

The `README` includes stuff from the original example repo as well. Ignore that
as I have changed a little bit from there. Follow the outlined steps. First,
clone the repo into a `ros2` ws. This would have a structure of the sort
`ws/src/{this repo}`. Next, Follow the singularity steps.

## Singularity

This is the docker type container to run to execute your code in simulation. Steps are as follows:
- Download the `singularity` debian file from [here](https://people.tuebingen.mpg.de/felixwidmaier/rrc2021/singularity.html)
- run `sudo apt install ./<.deb singularity file>`
- pull the challenge singularity image by running into your {this repo} folder.
```
singularity pull library://felix.widmaier/rrc/rrc2021:latest
```
- After that run `singularity build --fakeroot {sif file from previous step} ./image.def`. Make sure that the sif inside the `image.def` file points to the one retrieved from the previous step.

The previous steps should have built the required singularity container we need for running the stuff we have until now.

Next, launching the singularity container and running the pre-qual visualisation script.

```
export SINGULARITYENV_DISPLAY=$DISPLAY
singularity shell --cleanenv --no-home -B path/to/workspace path/to/rrc2021.sif
```
This should open a singularity shell. Make sure it points to the root of your workspace, i.e it should only have the `./src`

```
source /setup.bash
colcon build
```

This should build the workspace. You should have a `build`, `install`, `log` file in your workspace.

```
. ./install/local_setup.bash
```

You are now ready to run scripts in the package. To get started just launch the following and hopefully you can see the bots performing a trajectory


```
ros2 run rrc21 run_local_episode 4 mp-pg
```

# Stage 1 - Move Cube on Trajectory

## Workflow setup

**Note: Ensure that the previous steps are working**

The experimental workflow is as follows:
- develop your code and test on robot emulator simulation (different from the
  previous stage).
- commit your changes and push to remote git repo.
- `ssh` into the robot server and run your pushed code.

### Robot Emulator

The idea here is to sanity check your cod in a simulation before submitting it
to the actual bots

- If you have followed the previous steps, `singularity` should have been
  installed in your system. Ensure that you have python3 installed.
- install ROS2, prefereably `ros2-foxy` from [here](https://docs.ros.org/en/foxy/Installation.html).
- clone the emulator from [here](https://github.com/open-dynamic-robot-initiative/trifingerpro_runner), anywhere on your system. All of the code testing happens from here.

Run the following to setup your testing environment
```
export SINGULARITYENV_DISPLAY=$DISPLAY
source /opt/ros/foxy/setup.bash
```
Now, create a launch `launch.sh` script inside the `trifingerpro_runner` repo you just
cloned. Add the following inside. 
```
./run_simulation.py --output-dir ~/output \
                    --repository /path/to/your/experiment/code/.git \
                    --backend-image /path/to/.sif/file/XX.sif \
                    --task MOVE_CUBE_ON_TRAJECTORY \
		    --sim-visualize
```

Make sure your experiment code has been saved and committed as the script
uses the `HEAD` commit. Ensure that the output directory `~/output` is created.

You should see the output of your experiment's code in the `output` directory.

### Actual Robot

Once the code has been tested in the previous step:
- update the `branch` and `email` fields in the `roboch.json`
- copy the `roboch.json` to the remote robot server using
`scp roboch.json dopeytacos@robots.real-robot-challenge.com:`
you know where to find the password for this.
- you can then ssh into the main server and submit a job.

`ssh USERNAME@robots.real-robot-challenge.com`

Once logged in, type `submit` and job will be queued.

Ensure that your code is pushed to the remote repo before submitting.

## Collaboration points to keep in mind
Obviously this is a work in progress. So it would be better if we track issues
more systematically. Please raise an issue for any problem faced while trying
to run this code base, communicating the problem as succinctly as possible.
Let's coordinate experiments and job submission on slack.


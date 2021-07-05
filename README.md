# Stuff to do to get the code base running
The `README` includes stuff from the original example repo as well. Ignore that as I have changed a little bit from there. Follow the outlined steps. First, clone the repo into a `ros2` ws. This would have a structure of the sort `ws/src/{this repo}`. Next, Follow the singularity steps.

## Singularity
This is the docker type container to run to execute your code in simulation. Steps are as follows:
- Download the `singularity` debian file from [here](https://people.tuebingen.mpg.de/felixwidmaier/rrc2021/singularity.html)
- run `sudo apt install ./<.deb singularity file>`
- pull the challenge singularity image by running `$ singularity pull library://felix.widmaier/rrc/rrc2021:latest ` into your {this repo} folder.
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





================================================

** Ignore under this **





Example Package for the Real Robot Challenge 2021
=================================================

This is a basic example for a package that can be submitted to the robots of
the [Real Robot Challenge 2021](https://real-robot-challenge.com).

It is a normal ROS2 Python package that can be build with colcon.  However,
there are a few special files in the root directory that are needed for
running/evaluating your submissions.  See the sections on the different
challenge phases below for more on this.

This example uses purely Python, however, any package type that can be built
by colcon is okay.  So you can, for example, turn it into a CMake package if you
want to build C++ code.  For more information on this, see the [ROS2
documentation](https://docs.ros.org/en/foxy/Tutorials/Creating-Your-First-ROS2-Package.html).


Challenge Simulation Phase (Pre-Stage)
--------------------------------------

There are two example scripts using the simulation:

- `sim_move_up_and_down`:  Directly uses the `TriFingerPlatform` class to simply
  move the robot between two fixed positions.  This is implemented in
  `rrc_example_package/scripts/sim_move_up_and_down.py`.

- `sim_trajectory_example_with_gym`:  Wraps the robot class in a Gym environment
  and uses that to run a dummy policy which simply points with one finger on the
  goal positions of the trajectory.  This is implemented in 
  `rrc_example_package/scripts/sim_trajectory_example_with_gym.py`.

To execute the examples, [build the
package](https://people.tuebingen.mpg.de/felixwidmaier/rrc2021/singularity.html#singularity-build-ws)
and execute

    ros2 run rrc_example_package <example_name>


For evaluation of the pre-stage of the challenge, the critical file is the
`evaluate_policy.py` at the root directory of the package.  This is what is
going to be executed by `rrc_evaluate_prestage.py` (found in `scripts/`).

For more information, see the [challenge
documentation](https://people.tuebingen.mpg.de/felixwidmaier/rrc2021/)

`evaluate_policy.py` is only used for the simulation phase and not relevant
anymore for the later phases that use the real robot.


Challenge Real Robot Phases (Stages 1 and 2)
--------------------------------------------

For the challenge phases on the real robots, you need to provide the following
files at the root directory of the package such that your jobs can executed on
the robots:

- `run`:  Script that is executed when submitting the package to the robot.
  This can, for example, be a Python script or a symlink to a script somewhere
  else inside the repository.  In the given example, it is a shell script
  running a Python script via `ros2 run`.  This approach would also work for C++
  executables.  When executed, a JSON string encoding the goal is passed as
  argument (the exact structure of the goal depends on the current task).
- `goal.json`:  Optional.  May contain a fixed goal (might be useful for
  testing/training).  See the documentation of the challenge tasks for more
  details.

It is important that the `run` script is executable.  For this, you need to do
two things:

1. Add a shebang line at the top of the file (e.g. `#!/usr/bin/python3` when
   using Python or `#!/bin/bash` when using bash).
2. Mark the file as executable (e.g. with `chmod a+x run`).

When inside of `run` you want to call another script using `ros2 run` (as it is
done in this example), this other script needs to fulfil the same requirements.

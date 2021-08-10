source /setup.bash
colcon build
. ./install/local_setup.bash
export XDG_CONFIG_HOME=/home/aditya/Documents/projects/rrc_ws/rrc_example/src/rrc_example_package
echo $XDG_CONFIG_HOME

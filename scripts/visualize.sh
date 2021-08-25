# visualizes data from the experiments

# expects output directory and experiment number
if (( $# != 2))
then
	echo "Invalid number of arguments."
	echo "Usage: $0 <job_id> <output_directory"
	exit 1

fi

job_id=$1
output_dir=$2

experiment_dir="${output_dir}/${job_id}"

if ! [ -d "${experiment_dir}" ]
then
	echo "${experiment_dir} does not exist"
	exit 1
fi

# ./rrc2021_latest.sif python3 ./scripts/trifinger_platform_log_viewer.py "${experiment_dir}/robot_data.dat" "${experiment_dir}/camera_data.dat" -g "${experiment_dir}/goal.json"

# ./rrc2021_latest.sif ros2 run trifinger_object_tracking tricamera_log_converter \
#     "${experiment_dir}/camera_data.dat" "${experiment_dir}/video60.avi" -c camera60
# 
# 
# ./rrc2021_latest.sif ros2 run trifinger_object_tracking tricamera_log_converter \
#     "${experiment_dir}/camera_data.dat" "${experiment_dir}/video180.avi" -c camera180
# 
# ./rrc2021_latest.sif ros2 run trifinger_object_tracking tricamera_log_converter \
#     "${experiment_dir}/camera_data.dat" "${experiment_dir}/video300.avi" -c camera300
./rrc2021_latest.sif ros2 run trifinger_cameras tricamera_log_converter \
    "${experiment_dir}/camera_data.dat" "${experiment_dir}/video60.avi" -c camera60

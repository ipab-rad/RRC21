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

./rrc2021_latest.sif python3 ./scripts/trifinger_platform_log_viewer.py "${experiment_dir}/robot_data.dat" "${experiment_dir}/camera_data.dat" -g "${experiment_dir}/goal.json"

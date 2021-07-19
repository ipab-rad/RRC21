import os
import setuptools

PACKAGE_NAME = "rrc21"

setuptools.setup(
    name=PACKAGE_NAME,
    version="2.0.0",
    # Packages to export
    packages=setuptools.find_packages('submission'),
    package_dir={"": "submission"},
    data_files=[
        # Install "marker" file in package index
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + PACKAGE_NAME],
        ),
        # Include our package.xml file
        (os.path.join("share", PACKAGE_NAME), ["package.xml"]),
    ],
    # This is important as well
    install_requires=["setuptools"],
    zip_safe=True,
    author="Aditya Kamireddypalli",
    author_email="a.kamireddypalli@sms.ed.ac.uk",
    maintainer="Aditya Kamireddypalli",
    maintainer_email="a.kamireddypalli@sms.ed.ac.uk",
    description="Real Robot Challenge Submission from the RAD group, University of Edinburgh",
    license="BSD 3-clause",
    # Like the CMakeLists add_executable macro, you can add your python
    # scripts here.
    entry_points={
        "console_scripts": [
            "real_move_up_and_down = rrc_example_package.scripts.real_move_up_and_down:main",
            "sim_move_up_and_down = rrc_example_package.scripts.sim_move_up_and_down:main",
            "real_trajectory_example_with_gym = rrc_example_package.scripts.real_trajectory_example_with_gym:main",
            "real_trajectory_example_with_gym_dup = rrc_example_package.scripts.real_trajectory_example_with_gym_dup:main",
            "sim_trajectory_example_with_gym = rrc_example_package.scripts.sim_trajectory_example_with_gym:main",
            "dice_example_with_gym = rrc_example_package.scripts.dice_example_with_gym:main",
            "demo_gym = rrc_example_package.solution.demo_gym:main",
            "run_local_episode = rrc_example_package.solution.run_local_episode_rev:main",
        ],
    },
    package_data={
        "residual_learning": [
            "*.pt",
            "models/cic_lvl3/logs/*.gin",
            "models/cic_lvl4/logs/*.gin",
            "models/cpc_lvl3/logs/*.gin",
            "models/cpc_lvl4/logs/*.gin",
            "models/mp_lvl3/logs/*.gin",
            "models/mp_lvl4/logs/*.gin",
        ],
        "cic" : ["trifinger_mod.urdf"],
    },

)

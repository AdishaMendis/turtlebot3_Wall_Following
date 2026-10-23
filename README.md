# turtlebot3_Wall_Following
Project 2: TurtleBot3 Burger wall following using ROS 2 Humble and Gazebo Classic.

# PID wall-following simulation test

## Settings

- Robot: TurtleBot3 Burger
- Arena: 2 m x 2 m clear inside area
- Desired right-wall distance: 0.30 m
- Forward speed: 0.07 m/s
- Kp: 1.0
- Ki: 0.01
- Kd: 0.5
- Integral accumulation limit: +/-0.20 m.s
- Derivative filter time constant: 0.30 s
- Steering limit: +/-0.50 rad/s
- Corner trigger: front distance <=0.35 m
- Corner target: next fixed heading, 90 degrees left
- Corner tolerance: 2 degrees

## Recorded results

Source: pid_run_20261021_094836_461011.csv

- Wall-following samples: 517
- RMSE: 7.71 mm
- Mean absolute error: 6.14 mm
- Maximum absolute error: 21.38 mm
- Samples within +/-20 mm: 99.2%

Metrics describe logged wall-following samples.
Corner turns are excluded.
These results are from one simulation run.

## Run

Terminal 1:
    ros2 launch wall_follower simulation.launch.py

Terminal 2:
    ros2 run wall_follower scan_reader

Start the robot aligned with the first wall.
Press Ctrl+C in Terminal 2 to stop the controller.

CSV files are saved in:
    ~/wall_follower_logs/



source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run wall_follower scan_reader --ros-args \
  -p kp:=1.0 \
  -p ki:=0.01 \
  -p kd:=0.5 \
  -p forward_speed:=0.07 \
  -p desired_distance:=0.30
  
##  Single PID:
ros2 run wall_follower single_PID --ros-args \
  --params-file ~/ros2_ws/src/turtlebot3_Wall_Following/wall_follower/config/single_PID.yaml
  
## dual PID:
ros2 run wall_follower dual_PID --ros-args \
  --params-file ~/ros2_ws/src/turtlebot3_Wall_Following/wall_follower/config/dual_PID.yaml

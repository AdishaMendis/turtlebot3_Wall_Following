# ENG612 Project 2: TurtleBot3 Wall Following

## Objective
Use laser scans to maintain a target wall distance of 0.30 m
with a TurtleBot3 Burger.

Implement and compare:
1. Single PID for angular velocity, with constant or rule-based forward speed.
2. Dual PID for angular velocity and forward speed.

## Development environment
- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Classic
- TurtleBot3 Burger
- Python ROS package: wall_follower

## Repository structure
- wall_follower/wall_follower/: Python node source
- wall_follower/launch/: ROS launch files
- wall_follower/config/: controller parameters
- wall_follower/worlds/: Gazebo simulation world
- docs/: setup instructions and experiment notes

## Build
Run in a terminal:

    source /opt/ros/humble/setup.bash
    cd ~/ros2_ws
    colcon build --symlink-install --packages-select wall_follower
    source install/setup.bash

## Verify package discovery
    ros2 pkg prefix wall_follower

## Development status
Package skeleton only. Controller and simulation files will be
added in subsequent milestones.

## Collaboration
Both group members should commit their own contributions.
Use meaningful commit messages describing completed changes.
Do not commit generated build, install, or log directories.

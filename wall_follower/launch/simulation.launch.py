import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # Find the installed package folders.
    project = get_package_share_directory('wall_follower')
    gazebo = get_package_share_directory('gazebo_ros')
    turtlebot = get_package_share_directory('turtlebot3_gazebo')

    # Locate our arena world.
    world = os.path.join(project, 'worlds', 'wall_arena.world')

    # Open Gazebo with our arena.
    start_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world}.items(),
    )

    # Publish the robot's coordinate transforms.
    robot_state = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                turtlebot, 'launch', 'robot_state_publisher.launch.py'
            )
        ),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )

    # Place the robot inside the arena.
    spawn_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot, 'launch', 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': '-0.5',
            'y_pose': '-0.7',
        }.items(),
    )

    return LaunchDescription([
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),
        start_gazebo,
        robot_state,
        spawn_robot,
    ])

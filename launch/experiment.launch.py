#!/usr/bin/env python3
"""
BBAC ICS Framework - Experiment Launch
Launches BBAC system with evaluator node for experiments.
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for BBAC experiments."""
    
    # Package directory
    pkg_dir = get_package_share_directory('bbac_ics_core')
    
    # Configuration file
    config_file = os.path.join(pkg_dir, 'config', 'params.yaml')
    
    # Launch arguments
    config_arg = DeclareLaunchArgument(
        'config_file',
        default_value=config_file,
        description='Path to BBAC configuration file'
    )
    
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level'
    )
    
    experiment_name_arg = DeclareLaunchArgument(
        'experiment_name',
        default_value='baseline_test',
        description='Name of the experiment'
    )
    
    # Include system launch
    system_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, 'launch', 'system.launch.py')
        ),
        launch_arguments={
            'config_file': LaunchConfiguration('config_file'),
            'log_level': LaunchConfiguration('log_level'),
        }.items()
    )
    
    # Evaluator node
    evaluator_node = Node(
        package='bbac_ics_core',
        executable='evaluator_node',
        name='evaluator_node',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )
    
    # Launch description
    return LaunchDescription([
        config_arg,
        log_level_arg,
        experiment_name_arg,
        LogInfo(msg=['Launching BBAC Experiment: ', LaunchConfiguration('experiment_name')]),
        system_launch,
        evaluator_node,
    ])

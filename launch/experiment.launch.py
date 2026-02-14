#!/usr/bin/env python3
"""
BBAC_ICS Framework - Experiment Launch
Launches BBAC system with experiment evaluator.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for BBAC experiments."""
    
    # Package directory
    pkg_dir = get_package_share_directory('bbac_ics')
    
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
    
    ground_truth_arg = DeclareLaunchArgument(
        'ground_truth_file',
        default_value='ground_truth.json',
        description='Path to ground truth JSON'
    )
    
    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='results_experiment',
        description='Output directory for results'
    )
    
    # BBAC Main Node
    bbac_main_node = Node(
        package='bbac_ics',
        executable='bbac_main_node',
        name='bbac_main_node',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )
    
    # Experiment Evaluator Node
    evaluator_node = Node(
        package='bbac_ics',
        executable='experiment_evaluator_node',
        name='experiment_evaluator_node',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {'ground_truth_file': LaunchConfiguration('ground_truth_file')},
            {'output_dir': LaunchConfiguration('output_dir')}
        ],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )
    
    # Launch description
    return LaunchDescription([
        config_arg,
        log_level_arg,
        ground_truth_arg,
        output_dir_arg,
        LogInfo(msg=['Launching BBAC Experiment System...']),
        LogInfo(msg=['Output directory: ', LaunchConfiguration('output_dir')]),
        bbac_main_node,
        evaluator_node,
    ])

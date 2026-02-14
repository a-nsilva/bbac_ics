#!/usr/bin/env python3
"""
BBAC ICS Framework - System Launch
Launches all core BBAC nodes for production use.
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for BBAC system."""
    
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
        description='Logging level (debug, info, warn, error)'
    )
    
    # Nodes
    bbac_main_node = Node(
        package='bbac_ics_core',
        executable='bbac_main_node',
        name='bbac_main_node',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )
    
    baseline_manager_node = Node(
        package='bbac_ics_core',
        executable='baseline_manager_node',
        name='baseline_manager_node',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )
    
    # Launch description
    return LaunchDescription([
        config_arg,
        log_level_arg,
        LogInfo(msg=['Launching BBAC ICS System...']),
        LogInfo(msg=['Config file: ', LaunchConfiguration('config_file')]),
        bbac_main_node,
        baseline_manager_node,
    ])

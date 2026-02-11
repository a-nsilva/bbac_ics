#!/usr/bin/env python3
"""
BBAC ICS Framework - Baseline Manager Node
Periodically updates baselines and manages drift detection.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ..layers.behavioral_baseline import BaselineManager
from ..utils.config_loader import ConfigLoader


class BaselineManagerNode(Node):
    """Manages periodic baseline updates."""
    
    def __init__(self):
        super().__init__('baseline_manager_node')
        
        # Load configuration
        config = ConfigLoader.load()
        baseline_config = config.get('baseline', {})
        
        # Initialize baseline manager
        self.baseline_manager = BaselineManager(baseline_config)
        self.baseline_manager.load()
        
        # Periodic update timer (every 1 hour)
        update_interval = 3600.0  # seconds
        self.timer = self.create_timer(update_interval, self.periodic_update)
        
        # Publisher for status
        self.status_pub = self.create_publisher(
            String,
            '/bbac/baseline/status',
            10
        )
        
        self.get_logger().info(f"Baseline Manager Node initialized (update interval: {update_interval}s)")
    
    def periodic_update(self):
        """Periodic baseline maintenance."""
        self.get_logger().info("Running periodic baseline update...")
        
        # Save current baselines
        self.baseline_manager.save()
        
        # Publish status
        msg = String()
        msg.data = f"Baselines saved at {self.get_clock().now().to_msg()}"
        self.status_pub.publish(msg)
        
        self.get_logger().info("Baseline update complete")


def main(args=None):
    rclpy.init(args=args)
    node = BaselineManagerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

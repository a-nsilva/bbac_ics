#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

def main():
    import rclpy
    from rclpy.node import Node

    rclpy.init()

    node = MeuNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

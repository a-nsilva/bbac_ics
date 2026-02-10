#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class BBACMainNode(Node):
    def __init__(self):
        super().__init__('bbac_main_node')
        self.publisher_ = self.create_publisher(String, 'bbac_topic', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.count = 0
        self.get_logger().info("BBAC Main Node iniciado e publicando em 'bbac_topic'")

    def timer_callback(self):
        msg = String()
        msg.data = f"Mensagem {self.count}"
        self.publisher_.publish(msg)
        self.get_logger().info(f"Publicado: {msg.data}")
        self.count += 1

def main(args=None):
    rclpy.init(args=args)
    node = BBACMainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

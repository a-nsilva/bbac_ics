#!/usr/bin/env python3
"""
BBAC ICS Framework - Evaluator Node
Calculates real-time metrics for monitoring and evaluation.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from collections import defaultdict
import time

from bbac_ics_msgs.msg import AccessDecision as AccessDecisionMsg
from std_msgs.msg import String


class EvaluatorNode(Node):
    """Real-time metrics evaluation node."""
    
    def __init__(self):
        super().__init__('evaluator_node')
        
        # Metrics storage
        self.decisions = []
        self.latencies = []
        self.decision_counts = defaultdict(int)
        
        self.start_time = time.time()
        
        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscriber
        self.decision_sub = self.create_subscription(
            AccessDecisionMsg,
            '/bbac/decisions',
            self.process_decision,
            qos
        )
        
        # Publisher for metrics
        self.metrics_pub = self.create_publisher(
            String,
            '/bbac/metrics',
            10
        )
        
        # Periodic metrics report (every 60 seconds)
        self.timer = self.create_timer(60.0, self.report_metrics)
        
        self.get_logger().info("Evaluator Node initialized")
    
    def process_decision(self, msg: AccessDecisionMsg):
        """Process decision for metrics."""
        self.decisions.append(msg.decision)
        self.latencies.append(msg.latency_ms)
        self.decision_counts[msg.decision] += 1
    
    def report_metrics(self):
        """Report current metrics."""
        if not self.latencies:
            return
        
        # Compute metrics
        total_requests = len(self.decisions)
        elapsed_time = time.time() - self.start_time
        throughput = total_requests / elapsed_time if elapsed_time > 0 else 0
        
        avg_latency = sum(self.latencies) / len(self.latencies)
        p95_latency = sorted(self.latencies)[int(len(self.latencies) * 0.95)] if self.latencies else 0
        
        # Build report
        report = (
            f"=== BBAC Metrics Report ===\n"
            f"Total Requests: {total_requests}\n"
            f"Throughput: {throughput:.2f} req/s\n"
            f"Avg Latency: {avg_latency:.2f} ms\n"
            f"P95 Latency: {p95_latency:.2f} ms\n"
            f"Decision Breakdown:\n"
        )
        
        for decision, count in self.decision_counts.items():
            percentage = (count / total_requests) * 100
            report += f"  {decision}: {count} ({percentage:.1f}%)\n"
        
        self.get_logger().info(report)
        
        # Publish
        msg = String()
        msg.data = report
        self.metrics_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = EvaluatorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

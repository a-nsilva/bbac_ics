#!/usr/bin/env python3
"""
BBAC ICS Framework - Experiment Evaluator Node
Collects decisions, compares with ground truth, calculates metrics.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from collections import defaultdict
import time
import json
from pathlib import Path

from bbac_ics_msgs.msg import AccessDecision as AccessDecisionMsg
from std_msgs.msg import String


class ExperimentEvaluatorNode(Node):
    """Experiment evaluation with ground truth comparison."""
    
    def __init__(self):
        super().__init__('experiment_evaluator_node')
        
        # Declare parameters
        self.declare_parameter('ground_truth_file', '')
        self.declare_parameter('output_dir', 'results_experiment')
        
        # Get parameters
        gt_file = self.get_parameter('ground_truth_file').value
        self.output_dir = Path(self.get_parameter('output_dir').value)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load ground truth
        self.ground_truth = self._load_ground_truth(gt_file)
        self.get_logger().info(f"Loaded {len(self.ground_truth)} ground truth labels")
        
        # Results storage
        self.predictions = []
        self.ground_truth_labels = []
        self.latencies = []
        self.scores = []
        self.request_ids = []
        
        self.start_time = time.time()
        
        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100
        )
        
        # Subscriber
        self.decision_sub = self.create_subscription(
            AccessDecisionMsg,
            '/bbac/decisions',
            self.process_decision,
            qos
        )
        
        # Publisher for status
        self.status_pub = self.create_publisher(
            String,
            '/bbac/experiment/status',
            10
        )
        
        # Timer for periodic status (every 10 seconds)
        self.timer = self.create_timer(10.0, self.report_status)
        
        self.get_logger().info(f"Experiment Evaluator initialized (output: {self.output_dir})")
    
    def _load_ground_truth(self, filepath: str) -> dict:
        """Load ground truth from JSON file."""
        if not filepath:
            self.get_logger().warning("No ground truth file specified")
            return {}
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data  # {request_id: ground_truth_label}
        except Exception as e:
            self.get_logger().error(f"Failed to load ground truth: {e}")
            return {}
    
    def process_decision(self, msg: AccessDecisionMsg):
        """Process decision and compare with ground truth."""
        request_id = msg.request_id
        
        # Get ground truth
        if request_id not in self.ground_truth:
            self.get_logger().warning(f"No ground truth for {request_id}")
            return
        
        gt_label = self.ground_truth[request_id]
        
        # Convert decision to binary (1=allow, 0=deny)
        pred = 1 if msg.decision in ['allow', 'mfa'] else 0
        gt = 1 if gt_label in ['allow', 'grant'] else 0
        
        # Store
        self.request_ids.append(request_id)
        self.predictions.append(pred)
        self.ground_truth_labels.append(gt)
        self.latencies.append(msg.latency_ms)
        self.scores.append(msg.confidence)  # Use confidence as score
        
        # Progress
        if len(self.predictions) % 1000 == 0:
            self.get_logger().info(f"Processed: {len(self.predictions)} decisions")
    
    def report_status(self):
        """Report current status."""
        if not self.predictions:
            return
        
        total = len(self.predictions)
        elapsed = time.time() - self.start_time
        throughput = total / elapsed if elapsed > 0 else 0
        
        # Quick accuracy
        correct = sum(p == gt for p, gt in zip(self.predictions, self.ground_truth_labels))
        accuracy = correct / total if total > 0 else 0
        
        status = (
            f"Status: {total} processed | "
            f"Accuracy: {accuracy:.4f} | "
            f"Throughput: {throughput:.2f} req/s"
        )
        
        self.get_logger().info(status)
        
        # Publish
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
    
    def save_results(self):
        """Calculate final metrics and save to JSON."""
        if not self.predictions:
            self.get_logger().warning("No predictions to save")
            return
        
        self.get_logger().info("Calculating final metrics...")
        
        # Import metrics calculator
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        
        from bbac_ics_core.experiments.metrics_calculator import MetricsCalculator
        
        calc = MetricsCalculator()
        
        # Classification metrics
        metrics = calc.calculate_classification_metrics(
            self.ground_truth_labels,
            self.predictions,
            y_scores=self.scores
        )
        
        # Latency metrics
        latency_metrics = calc.calculate_latency_metrics(self.latencies)
        
        # Build results
        results = {
            'experiment_info': {
                'total_samples': len(self.predictions),
                'duration_seconds': time.time() - self.start_time,
                'throughput_req_per_sec': len(self.predictions) / (time.time() - self.start_time)
            },
            'metrics': {
                **metrics.to_dict(),
                'latency': latency_metrics.to_dict()
            },
            'predictions': self.predictions,
            'ground_truth': self.ground_truth_labels,
            'request_ids': self.request_ids
        }
        
        # Save to JSON
        output_file = self.output_dir / 'experiment_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.get_logger().info(f"Results saved to: {output_file}")
        
        # Print summary
        self.get_logger().info("=" * 50)
        self.get_logger().info("EXPERIMENT RESULTS")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Accuracy:  {metrics.accuracy:.4f}")
        self.get_logger().info(f"Precision: {metrics.precision:.4f}")
        self.get_logger().info(f"Recall:    {metrics.recall:.4f}")
        self.get_logger().info(f"F1 Score:  {metrics.f1:.4f}")
        self.get_logger().info(f"ROC-AUC:   {metrics.roc_auc:.4f}")
        self.get_logger().info(f"Avg Latency: {latency_metrics.mean:.4f} ms")
        self.get_logger().info("=" * 80)


def main(args=None):
    rclpy.init(args=args)
    node = ExperimentEvaluatorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Save results on shutdown
        node.save_results()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

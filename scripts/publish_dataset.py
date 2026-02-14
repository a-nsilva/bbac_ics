#!/usr/bin/env python3
"""
BBAC ICS - Dataset Publisher
Publishes test dataset to /bbac/requests topic.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bbac_ics_msgs.msg import AccessRequest as AccessRequestMsg
from bbac_ics_core.utils.data_loader import DataLoader
from bbac_ics_core.layers.ingestion import ingest_batch


class DatasetPublisher(Node):
    """Publishes dataset to ROS topic."""
    
    def __init__(self, dataset_path: str, ground_truth_output: str):
        super().__init__('dataset_publisher')
        
        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100
        )
        
        # Publisher
        self.publisher = self.create_publisher(
            AccessRequestMsg,
            '/bbac/requests',
            qos
        )
        
        # Load dataset
        self.get_logger().info(f"Loading dataset from: {dataset_path}")
        loader = DataLoader()
        loader.load_all()
        self.df = loader.load_split('test')
        self.df = ingest_batch(self.df)
        
        self.get_logger().info(f"Loaded {len(self.df)} samples")
        
        # Extract ground truth
        self.ground_truth = {}
        for idx, row in self.df.iterrows():
            request_id = str(row.get('log_id', f"req_{idx}"))
            gt_label = row.get('ground_truth', 'deny')
            self.ground_truth[request_id] = gt_label
        
        # Save ground truth to JSON
        gt_path = Path(ground_truth_output)
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gt_path, 'w') as f:
            json.dump(self.ground_truth, f, indent=2)
        
        self.get_logger().info(f"Ground truth saved to: {gt_path}")
    
    def publish_dataset(self, delay_ms: float = 1.0):
        """Publish all samples with delay."""
        self.get_logger().info("Starting dataset publication...")
        
        for idx, row in self.df.iterrows():
            # Convert to ROS message
            msg = self._row_to_msg(row, idx)
            
            # Publish
            self.publisher.publish(msg)
            
            # Progress
            if (idx + 1) % 1000 == 0:
                self.get_logger().info(f"Published: {idx + 1}/{len(self.df)}")
            
            # Delay
            time.sleep(delay_ms / 1000.0)
        
        self.get_logger().info(f"✓ All {len(self.df)} samples published")
    
    def _row_to_msg(self, row, idx) -> AccessRequestMsg:
        """Convert DataFrame row to ROS message."""
        msg = AccessRequestMsg()
        
        # Header
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Fields
        msg.request_id = str(row.get('log_id', f"req_{idx}"))
        msg.agent_id = str(row['agent_id'])
        msg.agent_type = str(row['agent_type'])
        msg.agent_role = str(row.get('robot_type', row.get('human_role', 'unknown')))
        msg.action = str(row['action'])
        msg.resource = str(row['resource'])
        msg.resource_type = str(row['resource_type'])
        msg.location = str(row['location'])
        msg.human_present = bool(row.get('human_present', False))
        msg.emergency = bool(row.get('emergency_flag', False))
        msg.session_id = str(row.get('session_id', ''))
        msg.previous_action = str(row.get('previous_action', ''))
        msg.auth_status = str(row.get('auth_status', 'success'))
        msg.attempt_count = int(row.get('attempt_count', 0))
        msg.priority = float(row.get('priority', 5.0))
        
        return msg


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Publish dataset to BBAC')
    parser.add_argument('--dataset', type=str, default='data/raw/test.csv',
                       help='Path to test dataset')
    parser.add_argument('--ground-truth-output', type=str, 
                       default='ground_truth.json',
                       help='Where to save ground truth')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between messages (ms)')
    
    args = parser.parse_args()
    
    rclpy.init()
    
    publisher = DatasetPublisher(
        dataset_path=args.dataset,
        ground_truth_output=args.ground_truth_output
    )
    
    # Wait for subscribers
    print("Waiting 3 seconds for subscribers...")
    time.sleep(3)
    
    # Publish
    publisher.publish_dataset(delay_ms=args.delay)
    
    # Cleanup
    publisher.destroy_node()
    rclpy.shutdown()
    
    print("✓ Dataset publication complete")


if __name__ == '__main__':
    main()

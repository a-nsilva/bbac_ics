#!/usr/bin/env python3
"""
BBAC ICS Framework - Main Orchestrator Node
Coordinates all layers and processes access requests.
"""

import time

import rclpy
from bbac_ics.msg import (
    AccessDecision as AccessDecisionMsg,
    AccessRequest as AccessRequestMsg,
    LayerDecisionDetail,
    LayerOutput,
)
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from bbac_ics_core.layers.authentication import AuthenticationModule
from bbac_ics_core.layers.baseline_manager import BaselineManager
from ..layers.decision_maker import DecisionMaker
from ..layers.feature_extractor import FeatureExtractor
from ..layers.fusion_layer import FusionLayer
from ..layers.ingestion import ingest_single
from ..layers.learning_updater import LearningUpdater
from ..layers.policy_engine import PolicyEngine
from ..models.sequence_predictor import SequencePredictor
from ..models.statistical_detector import StatisticalDetector
from ..utils.config_loader import ConfigLoader
from ..utils.data_structures import AccessDecision, AccessRequest


class BBACMainNode(Node):
    """Main BBAC orchestrator node."""
    
    def __init__(self):
        super().__init__('bbac_main_node')
        
        # Load configuration
        config = ConfigLoader.load()
        ros_config = config.get('ros', {})
        
        # QoS profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=ros_config.get('qos_depth', 10)
        )
        
        # Initialize layers
        self.get_logger().info("Initializing BBAC layers...")
        self._init_layers(config)
        
        # Publishers
        self.decision_pub = self.create_publisher(
            AccessDecisionMsg,
            '/bbac/decisions',
            qos
        )
        
        self.stat_output_pub = self.create_publisher(
            LayerOutput,
            '/bbac/layer/statistical',
            qos
        )
        
        self.seq_output_pub = self.create_publisher(
            LayerOutput,
            '/bbac/layer/sequence',
            qos
        )
        
        self.policy_output_pub = self.create_publisher(
            LayerOutput,
            '/bbac/layer/policy',
            qos
        )
        
        # Subscriber
        self.request_sub = self.create_subscription(
            AccessRequestMsg,
            '/bbac/requests',
            self.process_request,
            qos
        )
        
        self.get_logger().info("BBAC Main Node initialized")
    
    def _init_layers(self, config: dict):
        """Initialize all processing layers."""
        
        # Authentication
        thresholds = config.get('thresholds', {})
        self.auth_module = AuthenticationModule(
            max_attempts=thresholds.get('max_auth_attempts', 3)
        )
        
        # Baseline
        self.baseline_manager = BaselineManager(config.get('baseline'))
        self.baseline_manager.load()  # Load existing baselines
        
        # Feature extraction
        self.feature_extractor = FeatureExtractor(self.baseline_manager)
        
        # Models
        ml_config = config.get('ml_params', {})
        self.statistical_detector = StatisticalDetector(
            self.feature_extractor,
            anomaly_threshold=ml_config.get('statistical', {}).get('anomaly_threshold', 0.5)
        )
        
        self.sequence_predictor = SequencePredictor(
            sequence_length=ml_config.get('sequence', {}).get('sequence_length', 5),
            anomaly_threshold=ml_config.get('sequence', {}).get('anomaly_threshold', 0.5)
        )
        
        # Policy
        self.policy_engine = PolicyEngine(config.get('policy'))
        
        # Fusion
        self.fusion_layer = FusionLayer(config.get('fusion'))
        
        # Decision
        self.decision_maker = DecisionMaker(config.get('thresholds'))
        
        # Learning
        self.learning_updater = LearningUpdater(
            self.baseline_manager,
            self.sequence_predictor,
            config.get('learning')
        )
    
    def process_request(self, msg: AccessRequestMsg):
        """
        Process incoming access request.
        
        Args:
            msg: AccessRequest ROS message
        """
        start_time = time.time()
        
        # Convert ROS msg to AccessRequest
        request = self._msg_to_request(msg)
        
        self.get_logger().info(
            f"Processing request {request.request_id} from {request.agent_id}"
        )
        
        # Step 1: Authentication
        auth_valid, auth_reason = self.auth_module.authenticate(request)
        
        if not auth_valid:
            # Immediate denial
            decision = AccessDecision(
                request_id=request.request_id,
                timestamp=time.time(),
                decision="deny",
                confidence=1.0,
                latency_ms=(time.time() - start_time) * 1000,
                reason=auth_reason,
                layer_decisions={}
            )
            self._publish_decision(decision)
            return
        
        # Step 2: Ingestion (already done in msg conversion)
        
        # Step 3: Analysis layers (parallel)
        layer_decisions = {}
        
        # Statistical layer
        stat_decision = self.statistical_detector.analyze(request)
        layer_decisions['statistical'] = stat_decision
        self._publish_layer_output(stat_decision, self.stat_output_pub)
        
        # Sequence layer
        seq_decision = self.sequence_predictor.analyze(request)
        layer_decisions['sequence'] = seq_decision
        self._publish_layer_output(seq_decision, self.seq_output_pub)
        
        # Policy layer
        policy_decision = self.policy_engine.analyze(request)
        layer_decisions['policy'] = policy_decision
        self._publish_layer_output(policy_decision, self.policy_output_pub)
        
        # Step 4: Fusion
        hybrid_decision = self.fusion_layer.fuse(
            request.request_id,
            layer_decisions
        )
        
        # Step 5: Final decision
        final_decision = self.decision_maker.decide(request, hybrid_decision)
        
        # Step 6: Continuous learning
        self.learning_updater.process_decision(request, final_decision)
        
        # Publish decision
        self._publish_decision(final_decision)
        
        self.get_logger().info(
            f"Decision for {request.request_id}: {final_decision.decision} "
            f"(score={hybrid_decision.score:.3f}, latency={final_decision.latency_ms:.2f}ms)"
        )
    
    def _msg_to_request(self, msg: AccessRequestMsg) -> AccessRequest:
        """Convert ROS message to AccessRequest dataclass."""
        raw_dict = {
            'request_id': msg.request_id,
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'agent_id': msg.agent_id,
            'agent_type': msg.agent_type,
            'agent_role': msg.agent_role,
            'action': msg.action,
            'resource': msg.resource,
            'resource_type': msg.resource_type,
            'location': msg.location,
            'human_present': msg.human_present,
            'emergency': msg.emergency,
            'session_id': msg.session_id if msg.session_id else None,
            'previous_action': msg.previous_action if msg.previous_action else None,
            'auth_status': msg.auth_status,
            'attempt_count': msg.attempt_count,
            'priority': msg.priority,
        }
        
        return ingest_single(raw_dict)
    
    def _publish_layer_output(self, layer_decision, publisher):
        """Publish layer output to monitoring topic."""
        msg = LayerOutput()
        msg.layer_name = layer_decision.layer_name
        msg.score = layer_decision.score
        msg.confidence = layer_decision.confidence
        msg.metadata = [f"{k}={v}" for k, v in layer_decision.explanation.items()]
        
        publisher.publish(msg)
    
    def _publish_decision(self, decision: AccessDecision):
        """Publish final decision."""
        msg = AccessDecisionMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.request_id = decision.request_id
        msg.decision = decision.decision
        msg.confidence = decision.confidence
        msg.latency_ms = decision.latency_ms
        msg.reason = decision.reason
        msg.logged = True
        
        # Convert layer decisions
        if decision.layer_decisions:
            for layer_name, layer_data in decision.layer_decisions.items():
                detail = LayerDecisionDetail()
                detail.layer_name = layer_name
                detail.score = layer_data.get('score', 0.0)
                detail.decision = layer_data.get('decision', 'deny')
                detail.confidence = layer_data.get('confidence', 0.0)
                detail.latency_ms = layer_data.get('latency_ms', 0.0)
                
                # Serialize explanation to JSON
                import json
                detail.explanation_json = json.dumps(layer_data.get('explanation', {}))
                
                msg.layer_decisions.append(detail)
        
        self.decision_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BBACMainNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()





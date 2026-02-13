#!/usr/bin/env python3
"""
BBAC ICS Framework - Ablation Study
Evaluates impact of removing individual layers.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path
import json


from ..layers.decision_maker import DecisionMaker
from ..layers.baseline_manager import BaselineManager
from ..layers.ingestion import ingest_batch
from ..layers.feature_extractor import FeatureExtractor
from ..layers.fusion_layer import FusionLayer
from ..layers.policy_engine import PolicyEngine
from ..models.statistical_detector import StatisticalDetector
from ..models.lstm_predictor import LSTMPredictor
from ..utils.config_loader import ConfigLoader
from ..utils.data_structures import (
    ExperimentConfig,
    ExperimentResult,
    ClassificationMetrics
)
from ..utils.data_loader import DataLoader
from .metrics_calculator import MetricsCalculator


class AblationStudy:
    """
    Ablation study: systematically remove layers to measure impact.
    
    Configurations:
    1. Full system (all layers)
    2. No statistical layer
    3. No sequence layer
    4. No policy layer
    5. Policy only
    6. Statistical only
    7. Sequence only
    """
    
    def __init__(self, output_dir: str = "results/ablation"):
        """
        Initialize ablation study.
        
        Args:
            output_dir: Directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = ConfigLoader.load()
        self.metrics_calc = MetricsCalculator()
        
    def run(self, dataset_split: str = 'test') -> Dict:
        """
        Run ablation study on all configurations.
        
        Args:
            dataset_split: Dataset split to use ('train', 'validation', 'test')
            
        Returns:
            Results dictionary
        """
        print("=" * 60)
        print("ABLATION STUDY")
        print("=" * 60)
        
        # Load dataset
        data_loader = DataLoader()
        data_loader.load_all() 
        df = data_loader.load_split(dataset_split)
        df = ingest_batch(df)
        
        # Define configurations
        configs = [
            {'name': 'full_system', 'statistical': True, 'sequence': True, 'policy': True},
            {'name': 'no_statistical', 'statistical': False, 'sequence': True, 'policy': True},
            {'name': 'no_sequence', 'statistical': True, 'sequence': False, 'policy': True},
            {'name': 'no_policy', 'statistical': True, 'sequence': True, 'policy': False},
            {'name': 'policy_only', 'statistical': False, 'sequence': False, 'policy': True},
            {'name': 'statistical_only', 'statistical': True, 'sequence': False, 'policy': False},
            {'name': 'sequence_only', 'statistical': False, 'sequence': True, 'policy': False},
        ]
        
        results = {}
        
        for config in configs:
            print(f"\n{'-' * 60}")
            print(f"Configuration: {config['name']}")
            print(f"{'-' * 60}")
            
            result = self._run_configuration(df, config)
            results[config['name']] = result
            
            # Print summary
            metrics = result['metrics']
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"Latency (mean): {metrics['latency']['mean']:.2f} ms")
        
        # Save results
        self._save_results(results)
        
        # Generate comparison
        self._generate_comparison(results)
        
        return results
    
    def _run_configuration(self, df: pd.DataFrame, config: Dict) -> Dict:
        """Run single ablation configuration."""
        
        # Initialize components based on config
        baseline_manager = BaselineManager(self.config.get('baseline'))
        feature_extractor = FeatureExtractor(baseline_manager)
        
        statistical_detector = None
        if config['statistical']:
            statistical_detector = StatisticalDetector(
                feature_extractor,
                anomaly_threshold=0.5
            )
        
        lstm_predictor = None
        if config['sequence']:
            lstm_predictor = LSTMPredictor(
                sequence_length=5,
                anomaly_threshold=0.5
            )
        
        policy_engine = None
        if config['policy']:
            policy_engine = PolicyEngine(self.config.get('policy'))
        
        # Adjust fusion weights
        fusion_config = self.config.get('fusion', {}).copy()
        active_layers = sum([config['statistical'], config['sequence'], config['policy']])
        
        if active_layers == 0:
            raise ValueError("At least one layer must be enabled")
        
        # Redistribute weights
        weight_per_layer = 1.0 / active_layers
        fusion_config['weights'] = {
            'behavioral': weight_per_layer if config['statistical'] else 0.0,
            'ml': weight_per_layer if config['sequence'] else 0.0,
            'rule': weight_per_layer if config['policy'] else 0.0,
        }
        
        fusion_layer = FusionLayer(fusion_config)
        decision_maker = DecisionMaker(self.config.get('thresholds'))
        
        # Build baselines from training data
        data_loader = DataLoader()
        data_loader.load_all()
        train_df = data_loader.load_split('train')
        for agent_id in df['agent_id'].unique():
            agent_data = train_df[train_df['agent_id'] == agent_id]
            if not agent_data.empty:
                baseline_manager.compute_baseline(agent_id, agent_data)

        # Train MarkovChain if sequence layer enabled
        if lstm_predictor:
            from ..utils.data_utils import extract_session_sequences
            
            sequences = extract_session_sequences(
                train_df,
                session_col='session_id',
                action_col='action',
                min_length=2
            )
            sequence_list = list(sequences.values())
            
            if sequence_list:
                lstm_predictor.update_model(sequence_list)
                print(f"  ✓ MarkovChain trained with {len(sequence_list)} sequences")
            else:
                print(f"  ⚠ No sequences found for training")
        
        # Process requests
        predictions = []
        ground_truth = []
        latencies = []
        
        for idx, row in df.iterrows():
            request = self._row_to_request(row)
            
            # Analyze with active layers
            layer_decisions = {}
            
            if statistical_detector:
                stat_decision = statistical_detector.analyze(request)
                layer_decisions['statistical'] = stat_decision
            
            if lstm_predictor:
                seq_decision = lstm_predictor.analyze(request)
                layer_decisions['sequence'] = seq_decision
            
            if policy_engine:
                policy_decision = policy_engine.analyze(request)
                layer_decisions['policy'] = policy_decision
            
            # Fusion
            hybrid_decision = fusion_layer.fuse(request.request_id, layer_decisions)
            
            # Decision
            final_decision = decision_maker.decide(request, hybrid_decision)
            
            # Map decision to binary grant/deny
            pred = 1 if final_decision.decision in ['allow', 'mfa'] else 0
            gt = 1 if row.get('ground_truth', 'deny') in ['allow', 'grant'] else 0
            
            predictions.append(pred)
            ground_truth.append(gt)
            latencies.append(final_decision.latency_ms)
        
        # Calculate metrics with scores for ROC/PR curves
        # Convert predictions to scores (0-1 probabilities)
        # Como predictions são binários, usar latency como proxy ou fusion scores
        #metrics = self.metrics_calc.calculate_classification_metrics(
        #    ground_truth,
        #    predictions
        #)

        scores = [final_decision.confidence for final_decision in all_decisions]  # Precisa coletar
        
        # Por enquanto, usar predictions como scores (subótimo)
        metrics = self.metrics_calc.calculate_classification_metrics(
            ground_truth,
            predictions,
            y_scores=[p for p in predictions]  # Workaround: usa predições como scores
        )
        
        latency_metrics = self.metrics_calc.calculate_latency_metrics(latencies)
        
        return {
            'config': config,
            'metrics': {
                **metrics.to_dict(),
                'latency': latency_metrics.to_dict()
            },
            'predictions': predictions,
            'ground_truth': ground_truth
        }
    
    def _row_to_request(self, row):
        """Convert DataFrame row to AccessRequest."""
        from ..utils.data_structures import AccessRequest
        from ..layers.ingestion import ingest_single
        
        raw_dict = {
            'request_id': row.get('log_id', f"req_{row.name}"),
            'timestamp': row.get('timestamp', 0.0),
            'agent_id': row['agent_id'],
            'agent_type': row['agent_type'],
            'agent_role': row.get('robot_type', row.get('human_role', 'unknown')),
            'action': row['action'],
            'resource': row['resource'],
            'resource_type': row['resource_type'],
            'location': row['location'],
            'human_present': row.get('human_present', False),
            'emergency': row.get('emergency_flag', False),
            'auth_status': row.get('auth_status', 'success'),
            'attempt_count': row.get('attempt_count', 0),
        }
        
        return ingest_single(raw_dict)
    
    def _save_results(self, results: Dict):
        """Save ablation results to JSON."""
        output_file = self.output_dir / "ablation_results.json"
        
        # Prepare serializable results
        serializable = {}
        for config_name, result in results.items():
            serializable[config_name] = {
                'config': result['config'],
                'metrics': result['metrics']
            }
        
        with open(output_file, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    def _generate_comparison(self, results: Dict):
        """Generate comparison table."""
        print("\n" + "=" * 80)
        print("ABLATION COMPARISON")
        print("=" * 80)
        
        # Header
        print(f"{'Configuration':<20} {'Accuracy':<12} {'F1':<12} {'Latency (ms)':<15}")
        print("-" * 80)
        
        # Rows
        for config_name, result in results.items():
            metrics = result['metrics']
            print(
                f"{config_name:<20} "
                f"{metrics['accuracy']:<12.4f} "
                f"{metrics['f1_score']:<12.4f} "
                f"{metrics['latency']['mean']:<15.2f}"
            )


def main():
    """Run ablation study."""
    study = AblationStudy()
    results = study.run(dataset_split='test')


if __name__ == '__main__':
    main()










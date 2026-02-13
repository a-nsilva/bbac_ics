#!/usr/bin/env python3
"""
BBAC ICS Framework - Run All Experiments
Executes complete experimental evaluation.
"""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from bbac_ics_core.experiments.ablation_study import AblationStudy
from bbac_ics_core.experiments.adaptive_eval import AdaptiveEvaluation
from bbac_ics_core.experiments.dynamic_rules_test import DynamicRulesTest
from bbac_ics_core.utils.logger import setup_logger


logger = setup_logger('experiments', log_to_console=True)

def train_ensemble_model(output_dir: Path):
    """Train meta-classifier ensemble on training data."""
    logger.info("\n" + "=" * 80)
    logger.info("Training Ensemble Meta-Learner")
    logger.info("=" * 80)
    
    from bbac_ics_core.layers.fusion_layer import FusionLayer
    from bbac_ics_core.utils.data_loader import DataLoader
    from bbac_ics_core.layers.ingestion import ingest_batch
    from bbac_ics_core.layers.baseline_manager import BaselineManager
    from bbac_ics_core.layers.feature_extractor import FeatureExtractor
    from bbac_ics_core.layers.policy_engine import PolicyEngine
    from bbac_ics_core.models.statistical_detector import StatisticalDetector
    from bbac_ics_core.models.lstm_predictor import LSTMPredictor
    from bbac_ics_core.utils.config_loader import ConfigLoader
    
    config = ConfigLoader.load()
    
    # Load training data
    loader = DataLoader()
    loader.load_all()
    train_df = loader.load_split('train')
    train_df = ingest_batch(train_df)
    
    # Initialize components
    baseline_mgr = BaselineManager()
    feature_ext = FeatureExtractor(baseline_mgr)
    policy_engine = PolicyEngine()
    stat_detector = StatisticalDetector(feature_ext)
    lstm_predictor = LSTMPredictor()
    
    # Build baselines
    for agent_id in train_df['agent_id'].unique():
        agent_data = train_df[train_df['agent_id'] == agent_id]
        baseline_mgr.compute_baseline(agent_id, agent_data)
    
    # Collect layer scores
    layer_scores = []
    ground_truth = []
    
    for idx, row in train_df.iterrows():
        from bbac_ics_core.layers.ingestion import ingest_single
        request = ingest_single(row.to_dict())
        
        # Get layer scores
        policy_dec = policy_engine.analyze(request)
        stat_dec = stat_detector.analyze(request)
        seq_dec = lstm_predictor.analyze(request)
        
        layer_scores.append((
            policy_dec.score,
            stat_dec.score,
            seq_dec.score
        ))
        
        # Ground truth: 1=allow, 0=deny
        gt = 1 if row.get('ground_truth', 'deny') in ['allow', 'grant'] else 0
        ground_truth.append(gt)
    
    # Train ensemble
    fusion_config = config.get('fusion', {})
    fusion_config['use_meta_classifier'] = True
    fusion = FusionLayer(fusion_config)
    
    fusion.train_ensemble(layer_scores, ground_truth)
    
    # Save model
    model_dir = output_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    fusion.ensemble.save(model_dir / 'ensemble.pkl')
    
    # Evaluate
    metrics = fusion.ensemble.evaluate(layer_scores, ground_truth)
    logger.info(f"Ensemble Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Feature Importance: {metrics['feature_importance']}")
    
    return fusion
    
def run_all_experiments(output_dir: str = 'results', train_ensemble: bool = False):
    """
    Run complete experimental suite.
    
    Args:
        output_dir: Base directory for results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()

    # Train ensemble if requested
    if train_ensemble:
        try:
            train_ensemble_model(output_path)
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
    
    logger.info("=" * 80)
    logger.info("BBAC EXPERIMENTAL EVALUATION - FULL SUITE")
    logger.info("=" * 80)
    
    experiments = [
        ("Ablation Study", AblationStudy, 'ablation'),
        ("Adaptive Evaluation", AdaptiveEvaluation, 'adaptive'),
        ("Dynamic Rules Test", DynamicRulesTest, 'dynamic_rules'),
    ]
    
    results = {}
    
    for exp_name, exp_class, exp_dir in experiments:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Running: {exp_name}")
        logger.info(f"{'=' * 80}")
        
        try:
            exp_output = output_path / exp_dir
            experiment = exp_class(output_dir=str(exp_output))
            
            exp_start = time.time()
            result = experiment.run()
            exp_duration = time.time() - exp_start
            
            results[exp_name] = {
                'status': 'SUCCESS',
                'duration': exp_duration,
                'output_dir': str(exp_output)
            }
            
            logger.info(f"✓ {exp_name} completed in {exp_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"✗ {exp_name} failed: {e}")
            results[exp_name] = {
                'status': 'FAILED',
                'error': str(e)
            }
    
    total_duration = time.time() - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info("Generating Publication Plots")
    logger.info("=" * 80)
    
    try:
        from bbac_ics_core.utils.generate_plots import GeneratePlots
        import json
        
        plots = GeneratePlots(output_dir=output_path / 'figures')
        
        # 1. Ablation comparison
        ablation_file = output_path / 'ablation/ablation_results.json'
        if ablation_file.exists():
            with open(ablation_file) as f:
                ablation_data = json.load(f)
            plots.plot_ablation_comparison(
                ablation_data,
                filename='ablation_comparison.png'
            )
            logger.info("✓ Ablation comparison plot generated")
        
        # 2. Metrics comparison (se tiver múltiplas configs)
        # Extrair ClassificationMetrics de ablation_data
        from bbac_ics_core.utils.data_structures import ClassificationMetrics
        
        metrics_dict = {}
        for config_name, data in ablation_data.items():
            m = data['metrics']
            metrics_dict[config_name] = ClassificationMetrics(
                accuracy=m['accuracy'],
                precision=m['precision'],
                recall=m['recall'],
                f1=m['f1_score'],
                roc_auc=m.get('roc_auc', 0.0),
                avg_precision=m.get('average_precision', 0.0),
                tp=m['confusion_matrix']['tp'],
                tn=m['confusion_matrix']['tn'],
                fp=m['confusion_matrix']['fp'],
                fn=m['confusion_matrix']['fn']
            )
        
        plots.plot_metrics_comparison(
            metrics_dict,
            title='Ablation Study - Metrics Comparison',
            filename='metrics_comparison.png'
        )
        logger.info("✓ Metrics comparison plot generated")
        
        # 3. Latency distribution
        latencies_dict = {
            name: [data['metrics']['latency']['mean']] * 100  # Mock distribution
            for name, data in ablation_data.items()
        }
        
        plots.plot_latency_distribution(
            latencies_dict,
            title='Latency Distribution - All Configurations',
            filename='latency_distribution.png'
        )
        logger.info("✓ Latency distribution plot generated")

        # 4. Confusion matrices (top 3 configs)
        top_configs = ['full_system', 'statistical_only', 'policy_only']
        for config_name in top_configs:
            if config_name in metrics_dict:
                plots.plot_confusion_matrix(
                    metrics_dict[config_name],
                    title=f'Confusion Matrix - {config_name}',
                    filename=f'confusion_matrix_{config_name}.png'
                )
        logger.info(f"✓ Confusion matrices generated for {len(top_configs)} configs")
        
        # 5. ROC curves comparison
        if any(m.fpr and m.tpr for m in metrics_dict.values()):
            plots.plot_roc_curve(
                metrics_dict,
                title='ROC Curves - All Configurations',
                filename='roc_curves.png'
            )
            logger.info("✓ ROC curves generated")
        
        # 6. Precision-Recall curves
        if any(m.precision_curve and m.recall_curve for m in metrics_dict.values()):
            plots.plot_precision_recall_curve(
                metrics_dict,
                title='Precision-Recall Curves',
                filename='pr_curves.png'
            )
            logger.info("✓ PR curves generated")

        # 7. Adaptive drift plot (if data available)
        adaptive_file = output_path / 'adaptive/adaptive_results.json'
        if adaptive_file.exists():
            with open(adaptive_file) as f:
                adaptive_data = json.load(f)
            
            # A) Baseline convergence plot
            convergence_data = adaptive_data.get('convergence', {}).get('metrics', [])
            if convergence_data:
                window_sizes = [m['window_size'] for m in convergence_data]
                stability_scores = [1.0 - m['stability'] for m in convergence_data]
                
                plots.plot_adaptive_drift(
                    timestamps=window_sizes,
                    baseline_values=[0.8] * len(window_sizes),
                    current_values=stability_scores,
                    title='Baseline Convergence - Stability Over Window Size',
                    filename='baseline_convergence.png'
                )
                logger.info("✓ Baseline convergence plot generated")
            
            # B) Drift detection plot
            drift_data = adaptive_data.get('drift_adaptation', {})
            if 'action_drift' in drift_data:
                drift_score = drift_data['action_drift']
                
                plots.plot_drift_detection(
                    drift_score=drift_score,
                    threshold=0.15,
                    filename='drift_detection.png'
                )
                logger.info(f"✓ Drift detection plot generated (KL={drift_score:.4f})")
            
        plots.close_all()
        logger.info(f"✓ All plots saved to: {output_path / 'figures'}")
        
    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
        
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENTAL SUITE SUMMARY")
    logger.info("=" * 80)
    
    for exp_name, result in results.items():
        status = result['status']
        if status == 'SUCCESS':
            logger.info(f"✓ {exp_name}: {status} ({result['duration']:.2f}s)")
        else:
            logger.error(f"✗ {exp_name}: {status} - {result.get('error', 'Unknown error')}")
    
    logger.info(f"\nTotal execution time: {total_duration:.2f}s")
    logger.info(f"Results saved to: {output_path}")
    
    # Check if all succeeded
    all_success = all(r['status'] == 'SUCCESS' for r in results.values())
    
    if all_success:
        logger.info("\n✓ All experiments completed successfully!")
    else:
        logger.warning("\n⚠ Some experiments failed - check logs above")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run BBAC experiments')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()

    parser.add_argument(
        '--train-ensemble',
        action='store_true',
        help='Train ensemble meta-learner before experiments'
    )
    
    args = parser.parse_args()
    
    results = run_all_experiments(
        output_dir=args.output_dir,
        train_ensemble=args.train_ensemble
    )
    
    # Exit with error code if any experiment failed
    all_success = all(r['status'] == 'SUCCESS' for r in results.values())
    sys.exit(0 if all_success else 1)

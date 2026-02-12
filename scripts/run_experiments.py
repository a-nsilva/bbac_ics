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


def run_all_experiments(output_dir: str = 'results'):
    """
    Run complete experimental suite.
    
    Args:
        output_dir: Base directory for results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
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
    
    results = run_all_experiments(output_dir=args.output_dir)
    
    # Exit with error code if any experiment failed
    all_success = all(r['status'] == 'SUCCESS' for r in results.values())
    sys.exit(0 if all_success else 1)

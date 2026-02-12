#!/usr/bin/env python3
"""
BBAC ICS Framework - System Validation Script
Validates configuration, dependencies, and data files.
"""
import sys
from pathlib import Path
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bbac_ics_core.utils.config_loader import ConfigLoader
from bbac_ics_core.utils.logger import setup_logger


logger = setup_logger('validator', log_to_console=True)


class SystemValidator:
    """Validates BBAC system configuration and dependencies."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_all(self) -> bool:
        """Run all validation checks."""
        logger.info("=" * 60)
        logger.info("BBAC SYSTEM VALIDATION")
        logger.info("=" * 60)
        
        checks = [
            ("Configuration file", self.check_config),
            ("Python dependencies", self.check_dependencies),
            ("Data files", self.check_data_files),
            ("Message definitions", self.check_messages),
            ("Thresholds consistency", self.check_thresholds),
        ]
        
        for check_name, check_func in checks:
            logger.info(f"\n[Checking] {check_name}...")
            try:
                check_func()
                logger.info(f"✓ {check_name} - PASS")
            except Exception as e:
                self.errors.append(f"{check_name}: {str(e)}")
                logger.error(f"✗ {check_name} - FAIL: {e}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        if self.errors:
            logger.error(f"Errors found: {len(self.errors)}")
            for error in self.errors:
                logger.error(f"  - {error}")
        
        if self.warnings:
            logger.warning(f"Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        if not self.errors:
            logger.info("✓ All validation checks passed!")
            return True
        else:
            logger.error("✗ Validation failed - please fix errors above")
            return False
    
    def check_config(self):
        """Validate configuration file."""
        config_path = Path('config/params.yaml')
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load and validate structure
        config = ConfigLoader.load()
        
        required_sections = ['baseline', 'fusion', 'thresholds', 'learning', 'policy', 'ros', 'paths']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing config section: {section}")
        
        # Validate fusion weights sum to 1.0
        weights = config['fusion']['weights']
        total = sum(weights.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Fusion weights must sum to 1.0, got {total}")
    
    def check_dependencies(self):
        """Check Python dependencies."""
        required_packages = [
            'pandas',
            'numpy',
            'sklearn',
            'yaml',
            'matplotlib',
            'seaborn',
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            raise ImportError(f"Missing packages: {', '.join(missing)}")
    
    def check_data_files(self):
        """Check dataset files exist."""
        config = ConfigLoader.load()
        data_dir = Path(config['paths']['data_dir'])
        
        required_files = [
            config['paths']['train_file'],
            config['paths']['validation_file'],
            config['paths']['test_file'],
        ]
        
        missing = []
        for filename in required_files:
            filepath = data_dir / filename
            if not filepath.exists():
                missing.append(str(filepath))
        
        if missing:
            self.warnings.append(f"Missing data files: {', '.join(missing)}")
    
    def check_messages(self):
        """Check ROS message definitions."""
        msg_dir = Path('msg')
        
        required_msgs = [
            'AccessRequest.msg',
            'AccessDecision.msg',
            'LayerOutput.msg',
            'LayerDecisionDetail.msg',
            'EmergencyAlert.msg',
        ]
        
        missing = []
        for msg_file in required_msgs:
            if not (msg_dir / msg_file).exists():
                missing.append(msg_file)
        
        if missing:
            raise FileNotFoundError(f"Missing message files: {', '.join(missing)}")
    
    def check_thresholds(self):
        """Validate threshold ordering."""
        config = ConfigLoader.load()
        thresholds = config['thresholds']
        
        t_min = thresholds['t_min_deny']
        t1 = thresholds['t1_review']
        t2 = thresholds['t2_mfa']
        
        if not (t_min < t1 < t2):
            raise ValueError(
                f"Thresholds must be ordered: t_min_deny < t1_review < t2_mfa, "
                f"got {t_min} < {t1} < {t2}"
            )


def main():
    """Run validation."""
    validator = SystemValidator()
    success = validator.validate_all()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

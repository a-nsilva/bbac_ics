# BBAC ICS Framework

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)](https://github.com/a-nsilva/bbac_ics)

**Behavioral-Based Access Control Framework for Industrial Control Systems**

A hybrid access control system combining rule-based policies, behavioral analysis, and machine learning for adaptive security in ICS environments.

## Features

- **Tri-layer Hybrid Architecture**
  - Layer 1: Rule-based Access Control (RuBAC)
  - Layer 2: Behavioral Analysis (Markov chains)
  - Layer 3: ML Anomaly Detection (Isolation Forest)

- **Adaptive Learning**
  - Sliding window baseline (70% recent + 30% historical)
  - Continuous profile updates with trust filter
  - Drift detection and adaptation

- **ROS2 Integration**
  - Real-time decision making (sub-100ms target)
  - Multi-agent support (robots + humans)
  - Emergency alert system

## Installation

### Prerequisites

**System Requirements:**
- Ubuntu 22.04 LTS
- ROS2 Humble Hawksbill
- **Python 3.10** (strict requirement for ROS2 Humble compatibility)

### Setup

#### Clone and Setup Framework
```bash
# Create workspace
mkdir -p ~/bbac_ics
cd ~/bbac_ics

# Clone repository
git clone https://github.com/yourusername/bbac_ics.git
cd bbac_ics

# Important: This installs plotly and ensures correct numpy/scipy versions
```

#### Automatic Initialization
During container creation, the following steps are executed:
```bash
rosdep update
rosdep install --from-paths . --ignore-src -y
pip install -r requirements.txt
colcon build --symlink-install
```

#### Manual Build ROS2 Package (optional)
```bash
# Go to workspace root
/workspaces/bbac_ics

# Clean
rm -rf build install log

# Build package
colcon build --packages-select bbac_framework

# Source workspace
source install/setup.bash
```

#### Verify Installation
```bash
# Check if package is available
ros2 pkg list | grep bbac

# Should output: bbac_ics

# Verify Python imports
python3 -c "import numpy, scipy, pandas, sklearn, matplotlib, seaborn, plotly; print('All dependencies OK')"
```

## Usage

### Run BBAC Node
```bash
# Terminal 1: Launch BBAC node
ros2 run bbac_ics bbac_main_node.py

# Or with launch file (configurable parameters)
ros2 launch bbac_framework bbac.launch.py \
  enable_behavioral:=true \
  enable_ml:=true \
  enable_policy:=true
```

### Run Experiments
```bash
# Terminal 1: BBAC node (must be running)
ros2 run bbac_ics bbac_main_node.py

# Terminal 2: Run experiments sequentially
python3 bbac_ics_core/experiments/ablation_study.py
python3 bbac_ics_core/experiments/adaptative_eval.py
python3 bbac_ics_core/experiments/dynamic_rules.py
```

## Configuration

Edit YAML files in `config/params.yaml`:

- Baseline window settings (sliding window: 70% recent + 30% historical)
- Layer fusion weights (default: rule = 0.4, behavioral = 0.3, ml = 0.3)
- Decision thresholds (T1 = 0.7 grant, T2 = 0.5 MFA, T3 = 0.3 review)
- System parameters (target latency: 100ms)

## Architecture
```
bbac_ics/
├── .devcontainer/
│   ├── devcontainer.json
│   └── Dockerfile
├── bbac_ics_core/
│   ├── __init__.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── ablation_study.py
│   │   ├── adaptive_eval.py
│   │   └── dynamic_rules.py
│   ├── layers/     
│   │   ├── __init__.py
│   │   ├── authentication.py
│   │   ├── behavioral.py    # Layer 3: Statistical + Sequence + Policy
│   │   ├── decision.py      # Layer 4b: Risk classification + RBAC
│   │   ├── feature_extractor.py
│   │   ├── fusion.py        # Layer 4a: Score fusion
│   │   ├── ingestion.py     # Layer 1: Auth + preprocessing
│   │   ├── learning.py      # Layer 5: Continuous learning
│   │   └── modeling.py      # Layer 2: Baseline + profiles
│   ├── models/      # LSTM, statistical, fusion
│   │   └── __init__.py
│   ├── nodes/         # ROS2 integration
│   │   ├── __init__.py
│   │   ├── baseline_manager_node.py
│   │   ├── bbac_main_node.py
│   │   └── evalutior_node.py
│   └── util/        # Utilities
│       ├── --init__.py
│       ├── config_loader.py
│       ├── data_loader.py
│       ├── data_structures.py
│       ├── data_utils.py
│       ├── generate_plots.py
│       └── logger.py
├── config/        
│   └── params.yaml
├── data/            # Dataset (1m samples)
│   ├── processed/
│   └── raw/
├── launch/          # ROS2 launch files
├── msg/             # ROS2 custom messages
└── tests/           # Unit tests
```

## Architecture
-  system follows a layered architecture:
- Ingestion Layer
- Feature Extraction
- Behavioral Baseline
- Statistical Detection
- ML Prediction
- Fusion Layer
- Policy Engine (RuBAC)
- Decision Maker

Each layer is implemented as a modular component.

## Experiments

### Ablation Study
Tests individual layers vs hybrid approach:
- Rule-only (RuBAC)
- Statistical-only
- Sequence-only (Markov)
- Statistical + Sequence
- **Full hybrid** (all layers)

**Results:** `results/ablation_study/`

### Baseline Comparison
Compares BBAC against traditional methods:
- RBAC (Role-Based)
- ABAC (Attribute-Based)
- Rule-based only
- Behavioral-only
- **BBAC** (hybrid)

**Results:** `results/baseline_comparison/`

### Adaptive Evaluation
Evaluates 6 key ADAPTIVE/DYNAMIC metrics:

**ADAPTIVE:**
1. Baseline convergence rate
2. Drift detection accuracy
3. Sliding window effectiveness (70/30 validation)

**DYNAMIC:**
4. Rule update latency (<1s target)
5. Conflict resolution rate (100% target)

**INTERACTION:**
6. Concurrent drift + rule change handling

**Results:** `results/adaptive_evaluation/`

## Results

All results saved to `results/` with publication-quality figures:

## Python Version Compatibility

**Critical:** This framework requires **Python 3.10 exactly** due to ROS2 Humble constraints.

Package versions are carefully selected to avoid conflicts:
- `numpy <1.25.0` - Compatible with scipy/sklearn/tensorflow
- `scipy <1.12.0` - Avoids apt package conflicts
- `scikit-learn ~=1.3.0` - Allows patches (1.3.x) but not 1.4.x

**Do not upgrade** to Python 3.11+ or newer package versions without testing ROS2 compatibility.

## Troubleshooting

### Import Errors
```bash
# If you get numpy/scipy import errors:
pip3 uninstall numpy scipy scikit-learn
pip3 install -r requirements.txt
```

### ROS2 Node Not Found
```bash
# Re-source workspace
source ~/bbac_ws/install/setup.bash

# Rebuild if needed
cd ~/bbac_ws
colcon build --packages-select bbac_ics
```

### Permission Denied on Scripts
```bash
# Make scripts executable
chmod +x experiments/*.py
chmod +x src/ros/bbac_main_node.py
```

## Citation

If you use this framework in your research, please cite:
```bibtex
@article{yourname2026bbac,
  title={BBAC: Behavioral-Based Access Control for Industrial Control Systems},
  author={Your Name and Coauthor Name},
  journal={Journal Name},
  year={2026},
  note={Target IF>7}
}
```

## License

APACHE 2.0 License - see LICENSE file for details.

## Contact

- Author: Your Name (your.email@example.com)
- GitHub: https://github.com/yourusername/bbac-framework
- Issues: https://github.com/yourusername/bbac-framework/issues

## Acknowledgments

- ROS2 Humble community
- scikit-learn, numpy, scipy contributors
- Research institution name

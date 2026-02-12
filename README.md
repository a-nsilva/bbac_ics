# BBAC ICS Framework

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)](https://github.com/a-nsilva/bbac_ics)

**Behavioral-Based Access Control Framework for Industrial Control Systems**

A hybrid access control system combining Rule-based and Behavior-Attribute Access Control (RuBAC), adaptive behavioral analysis, and machine learning for security in ICS environments.

## Features

- **Multi-Layer Hybrid Architecture (8 layers)**
  - **Layer 1:** Authentication validation
  - **Layer 2:** Data ingestion and preprocessing  
  - **Layer 3:** Adaptive baseline (70% recent + 30% historical)
  - **Layer 4:** Feature extraction
  - **Layer 5:** Triple analysis engine (Policy + Behavioral + ML in parallel)
    - Policy Engine: RuBAC policies with dynamic updates
    - Statistical Detector: Anomaly detection via behavioral baseline
    - LSTM Predictor: Sequence-based pattern analysis
  - **Layer 6:** Score fusion with configurable weights
  - **Layer 7:** Decision making with multi-threshold logic
  - **Layer 8:** Continuous learning from trusted samples

- **Adaptive Learning**
  - Sliding window baseline with weighted merging
  - Continuous profile updates from trusted samples
  - Drift detection and automatic adaptation

- **Real-time Decision Making**
  - Target latency: < 100ms
  - Score fusion with configurable weights
  - Multi-threshold decision logic (allow, MFA, review, deny)

- **ROS2 Integration**
  - Multi-agent support (robots + humans)
  - Distributed processing via topics
  - Real-time metrics evaluation

## Installation

### Prerequisites

**System Requirements:**
- Ubuntu 22.04 LTS
- ROS2 Humble Hawksbill
- **Python 3.10** (strict requirement for ROS2 Humble compatibility)

### Setup

#### 1. Clone Repository
```bash
mkdir -p ~/bbac_ws/src
cd ~/bbac_ws/src
git clone https://github.com/a-nsilva/bbac_ics.git
cd bbac_ics
```

#### 2. Install Dependencies
```bash
# ROS dependencies
rosdep update
rosdep install --from-paths . --ignore-src --rosdistro humble -y

# Python dependencies (carefully versioned for compatibility)
pip3 install -r requirements.txt
```

#### 3. Build ROS2 Package
```bash
cd ~/bbac_ws
colcon build --symlink-install
source install/setup.bash
```

#### 4. Verify Installation
```bash
# Check ROS package
ros2 pkg list | grep bbac_ics

# Verify Python imports
python3 -c "import numpy, scipy, pandas, sklearn, matplotlib, seaborn, plotly; print('✓ All dependencies OK')"

# Validate system
python scripts/validate_system.py
```

## Usage

### Running Experiments (For Research Paper)

**This is what you need to generate paper results:**
```bash
# 1. Validate configuration (optional but recommended)
python scripts/validate_system.py

# 2. Run all experiments (generates metrics, tables, figures)
python scripts/run_experiments.py --output-dir results
```

**Results saved to:**
- `results/ablation/` - Ablation study (layer impact analysis)
- `results/adaptive/` - Adaptive baseline evaluation
- `results/dynamic_rules/` - Dynamic rule update tests
- `results/figures/` - Publication-quality plots

**Run individual experiments:**
```bash
# Ablation study
ablation_study

# Adaptive evaluation
adaptive_eval

# Dynamic rules test
dynamic_rules_test
```

---

### Running ROS System (Optional - For Demonstration)

**Use this only to validate ROS integration, NOT to generate paper results:**
```bash
# Terminal 1: Launch full system
ros2 launch bbac_ics_core system.launch.py

# Terminal 2: Launch with evaluator (real-time metrics)
ros2 launch bbac_ics_core experiment.launch.py
```

**What this does:**
- Starts ROS nodes for real-time decision processing
- Subscribes to `/bbac/requests` topic
- Publishes decisions to `/bbac/decisions` topic
- Evaluates latency and throughput in real-time

**Note:** ROS launch is for production/demonstration purposes. Paper experiments run offline on CSV datasets.

---

## Configuration

Edit `config/params.yaml` to customize:

### Baseline Configuration
```yaml
baseline:
  window_days: 10
  recent_weight: 0.7        # 70% recent data
  # historical_weight: 0.3  # 30% historical (auto-calculated)
```

### Fusion Weights
```yaml
fusion:
  weights:
    rule: 0.4               # Policy layer
    behavioral: 0.3         # Statistical layer
    ml: 0.3                 # Sequence layer
```

### Decision Thresholds
```yaml
thresholds:
  # Score represents legitimacy/confidence (higher = safer)
  t_min_deny: 0.2          # score < 0.2 → auto deny + alert
  t1_review: 0.4           # 0.2 ≤ score < 0.4 → manual review
  t2_mfa: 0.6              # 0.4 ≤ score < 0.6 → MFA required
  # score ≥ 0.6 → allow
```

### Learning Parameters
```yaml
learning:
  buffer_size: 1000               # Samples before update
  trust_threshold: 0.8            # Minimum confidence
  min_samples_for_update: 100
```

---

## Architecture
```
bbac_ics/
├── .devcontainer/
│   ├── devcontainer.json       # VS Code Dev Container config
│   └── Dockerfile              # ROS2 Humble + Python environment
├── bbac_ics_core/              # Main package
│   ├── experiments/            # Evaluation framework
│   │   ├── ablation_study.py           # Layer impact analysis
│   │   ├── adaptive_eval.py            # Baseline adaptation tests
│   │   ├── dynamic_rules_test.py       # Rule update performance
│   │   └── metrics_calculator.py       # Centralized metrics
│   ├── layers/                 # Processing layers
│   │   ├── authentication.py           # Auth validation
│   │   ├── behavioral_baseline.py      # Adaptive baseline (70/30)
│   │   ├── decision_maker.py           # Final decision logic
│   │   ├── feature_extractor.py        # Feature engineering
│   │   ├── fusion_layer.py             # Score fusion
│   │   ├── ingestion.py                # Data preprocessing
│   │   ├── learning_updater.py         # Continuous learning
│   │   └── policy_engine.py            # RuBAC policies
│   ├── models/                 # ML models
│   │   ├── lstm_predictor.py           # Sequence prediction
│   │   └── statistical_detector.py     # Anomaly detection
│   ├── nodes/                  # ROS2 nodes
│   │   ├── bbac_main_node.py           # Main orchestrator
│   │   ├── baseline_manager_node.py    # Baseline updates
│   │   └── evaluator_node.py           # Real-time metrics
│   └── utils/                  # Utilities
│       ├── config_loader.py            # YAML config
│       ├── data_loader.py              # Dataset loading
│       ├── data_structures.py          # Dataclasses & enums
│       ├── data_utils.py               # Data processing helpers
│       ├── generate_plots.py           # Publication plots
│       └── logger.py                   # Logging config
├── config/
│   └── params.yaml             # System configuration
├── data/
│   └── raw/                    # Dataset (train/val/test CSVs)
├── launch/                     # ROS2 launch files
│   ├── system.launch.py        # Production system
│   └── experiment.launch.py    # With evaluator
├── msg/                        # ROS2 message definitions
│   ├── AccessRequest.msg
│   ├── AccessDecision.msg
│   ├── EmergencyAlert.msg
│   ├── LayerOutput.msg
│   └── LayerDecisionDetail.msg
├── scripts/                    # Standalone scripts
│   ├── validate_system.py      # System validation
│   └── run_experiments.py      # Run all experiments
└── requirements.txt            # Python dependencies
```

---

## Experiments

### 1. Ablation Study

Tests individual layers vs hybrid approach to measure each layer's contribution:

**Configurations tested:**
- Full system (all layers)
- No statistical layer
- No sequence layer
- No policy layer
- Policy only
- Statistical only
- Sequence only

**Metrics:**
- Accuracy, Precision, Recall, F1
- ROC-AUC
- Latency (mean, p95, p99)

**Output:** `results/ablation/ablation_results.json`

---

### 2. Adaptive Evaluation

Evaluates adaptive and dynamic capabilities:

**Adaptive Metrics:**
1. **Baseline convergence rate** - How quickly baseline stabilizes
2. **Sliding window effectiveness** - Validates 70/30 weighting
3. **Drift adaptation** - Response to behavioral changes

**Dynamic Metrics:**
4. **Rule update latency** - Target: < 1 second
5. **Rule consistency** - Target: > 99.9% during transitions

**Output:** `results/adaptive/adaptive_results.json`

---

### 3. Dynamic Rules Test

Tests policy engine rule updates:

**Metrics:**
- Rule update latency (target: < 1000ms)
- Consistency during transitions (target: > 99.9%)
- Conflict detection

**Output:** `results/dynamic_rules/dynamic_rules_results.json`

---

## Dataset

Place your dataset files in `data/raw/`:
```
data/raw/
├── trainer.csv        # Training data
├── validation.csv     # Validation data
└── test.csv          # Test data (for experiments)
```

**Required columns:**
- `log_id`, `timestamp`, `session_id`
- `agent_id`, `agent_type`, `robot_type`/`human_role`
- `action`, `resource`, `resource_type`
- `location`, `human_present`, `emergency_flag`
- `previous_action`, `auth_status`, `attempt_count`
- `ground_truth` (for evaluation)

---

## Python Version Compatibility

**Critical:** This framework requires **Python 3.10 exactly** due to ROS2 Humble constraints.

Package versions are carefully selected to avoid conflicts:
- `numpy <1.25.0` - Compatible with scipy/sklearn/tensorflow
- `scipy <1.12.0` - Avoids apt package conflicts
- `scikit-learn ~=1.3.0` - Allows patches (1.3.x) but not 1.4.x

**Do not upgrade** to Python 3.11+ or newer package versions without testing ROS2 compatibility.

---

## Troubleshooting

### Import Errors
```bash
# If you get numpy/scipy import errors:
pip3 uninstall numpy scipy scikit-learn -y
pip3 install -r requirements.txt
```

### ROS2 Package Not Found
```bash
# Re-source workspace
source ~/bbac_ws/install/setup.bash

# Rebuild if needed
cd ~/bbac_ws
colcon build --symlink-install
```

### Experiments Not Found
```bash
# Verify entry points
pip3 show bbac-ics-core

# Reinstall if needed
cd ~/bbac_ws
colcon build --symlink-install
source install/setup.bash
```

### Permission Denied
```bash
# Make scripts executable
chmod +x scripts/*.py
```

---

## Citation

If you use this framework in your research, please cite:
```bibtex
@article{silva2026bbac,
  title={BBAC: Adaptive Behavior-Based Access Control for Industrial Control Systems},
  author={Silva, Alexandre do Nascimento and Coauthor Name},
  journal={IEEE Transactions on [Target Journal]},
  year={2026},
  note={Submitted}
}
```

---

## 📜 License

Apache License 2.0 - see LICENSE file for details.

---

## Contact

- **Author:** Alexandre do Nascimento Silva
- **Email:** alnsilva@uesc.br
- **GitHub:** https://github.com/a-nsilva/bbac_ics
- **Issues:** https://github.com/a-nsilva/bbac_ics/issues

---

## 👥 Authors & Contact

- **Nastaran Farhadighalati**
  Nova University Lisbon (UNINOVA) Center of Technology and Systems (CTS), Department of Electrical Engineering and Computer, School of Science and Technology. Foundation for Science and Technology (FCT)
  Associated Lab of Intelligent Systems (LASI)

- **Alexandre do Nascimento Silva** (Corresponding Author)  
  Universidade Estadual de Santa Cruz (UESC), Departamento de Engenharias e Computação
  Universidade do Estado da Bahia (UNEB), Programa de Pós-graduação em Modelagem e Simulação em Biossistemas (PPGMSB)
  📧 monteiro.br

- **Roberto Luiz Souza Monteiro** (Corresponding Author)  
  Universidade SENAI-CIMATEC, 
  Universidade do Estado da Bahia (UNEB), Programa de Pós-graduação em Modelagem e Simulação em Biossistemas (PPGMSB)
  📧 roberto.monteiro@fieb.org.br

- **Sanaz Nikghadam-Hojjati**  
  Nova University Lisbon (UNINOVA) Center of Technology and Systems (CTS), Department of Electrical Engineering and Computer, School of Science and Technology. Foundation for Science and Technology (FCT)
  Associated Lab of Intelligent Systems (LASI)

- **José Barata**  
  Nova University Lisbon (UNINOVA) Center of Technology and Systems (CTS), Department of Electrical Engineering and Computer, School of Science and Technology. Foundation for Science and Technology (FCT)
  Associated Lab of Intelligent Systems (LASI)

- **Luiz Estrada**  
  Nova University Lisbon (UNINOVA) Center of Technology and Systems (CTS), Department of Electrical Engineering and Computer, School of Science and Technology. Foundation for Science and Technology (FCT)
  Associated Lab of Intelligent Systems (LASI)

## 🙏 Acknowledgments

This research was supported by:
- UNINOVA—Center of Technology and Systems (CTS)
- Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES)
- Universidade SENAI-CIMATEC
- Universidade Estadual de Santa Cruz (UESC)
- Universidade do Estado da Bahia (UNEB)

---

**Last Updated**: February 2026  
**Repository Status**: Under active development for publication 

\address[3]{, and , , Monte de Caparica, 2829-516, Portugal} 

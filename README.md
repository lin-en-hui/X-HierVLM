# X-HierVLM: Interpretable Hierarchical Image Classification

[![Paper](https://img.shields.io/badge/Paper-📄-blue)](https://github.com/lin-en-hui/X-HierVLM)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)

Official implementation of the paper:

**"An Interpretable Hierarchical Image Classification Framework Fusing Domain Priors and Pre-trained Vision-Language Models"**

Enhui Lin, Hainan Chen, Xiaoan Bao, Zhuang Zheng, Ruohan Wang, Jiajun Wang

---

## 📖 Overview

X-HierVLM is a dual-track framework for fine-grained visual categorization that combines:

- **White-box attribute reasoning**: A fine-tuned VLM (Qwen2.5-VL) with hierarchical attribute ontology, ICF weighting, and DAG-constrained path search, producing traceable evidence chains.
- **Black-box domain prior**: BioCLIP 2.5 providing robust fine-grained features for challenging cases (occlusion, camouflage).
- **Dual-track arbitration**: Baseline calibration strategy based on historical precision and average correct margin.

### Key Features

- 🧩 **Hierarchical reasoning** with structured visual attributes (L1~L6)
- 📊 **Comprehensive metrics**: Acc_L1~L6, HCA, EPR, POR, S-POR, TOR, ECC, EPC
- 🔍 **Interpretable decisions** with explicit evidence chains
- ⚖️ **Dual-track arbitration** balancing accuracy and transparency
- 🚀 **Batch inference** support for CSV and directory modes

---

## 🔧 Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/lin-en-hui/X-HierVLM.git
cd X-HierVLM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

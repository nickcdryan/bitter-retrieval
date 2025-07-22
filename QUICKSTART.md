# Bitter Retrieval - Quick Start Guide

A simplified training pipeline for retrieval models with LLM-guided signals.

## 🚀 Quick Setup

### Prerequisites
- Fresh Linux server (Ubuntu/Debian/Fedora/RHEL/Arch supported)
- Python 3.8+ (will be installed if missing)
- CUDA-capable GPU (recommended)
- Your soft-labeled data on Google Drive

### 1. Automated Setup (Recommended for New Servers)
For a completely fresh Linux server, run our automated setup:
```bash
git clone <your-repo-url>
cd bitter-retrieval
chmod +x setup.sh
./setup.sh
```

This script will:
- ✅ Detect your Linux distribution
- ✅ Install system dependencies (Python, curl, git, etc.)
- ✅ Install pipx and Poetry
- ✅ Configure PyTorch with CUDA 12.8 support
- ✅ Install all ML dependencies (transformers, datasets, wandb)
- ✅ Set up helpful aliases and environment
- ✅ Verify PyTorch CUDA installation

### 2. Manual Setup (If You Already Have Poetry)
```bash
# Add PyTorch CUDA source
poetry source add --priority=explicit pytorch https://download.pytorch.org/whl/cu128

# Install dependencies
poetry install
```

### 3. Setup API Keys
```bash
poetry run python setup_env.py
```
This will guide you through setting up your Hugging Face and Weights & Biases API keys.

### 4. Verify Setup
```bash
poetry run python test_setup.py
```
This will verify that all dependencies are installed and CUDA is working correctly.

### 5. Download Your Data
Get your Google Drive file ID from the share link:
```
https://drive.google.com/file/d/1ABCdef123xyz456/view
                                   ↑ This is your file ID
```

Download your data:
```bash
python download_data.py YOUR_FILE_ID_HERE
```

### 3. Start Training
```bash
python -m src.bitter_retrieval.train --method standard_infonce
```

## 🎯 Training Methods

- **`standard_infonce`** - Standard InfoNCE with hard labels
- **`converted_infonce`** - InfoNCE with LLM-converted labels  
- **`kl_soft_infonce`** - KL divergence with soft LLM distributions

## 📋 Command Line Options

### Basic Usage
```bash
python -m src.bitter_retrieval.train --method standard_infonce
```

### Common Parameters
```bash
python -m src.bitter_retrieval.train \
    --method standard_infonce \
    --learning-rate 2e-5 \
    --batch-size 2 \
    --num-epochs 2 \
    --temperature 0.02 \
    --device cuda \
    --seed 42
```

### Using Config Files
```bash
python -m src.bitter_retrieval.train --config my_config.json
```

Example config (`my_config.json`):
```json
{
  "method": "standard_infonce",
  "learning_rate": 2e-5,
  "batch_size": 2,
  "num_epochs": 2,
  "temperature": 0.02,
  "train_size": 25000,
  "use_wandb": true,
  "device": "cuda"
}
```

### Override Config Settings
```bash
python -m src.bitter_retrieval.train --config my_config.json --batch-size 4 --num-epochs 3
```

## 🔧 Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `standard_infonce` | Training method |
| `learning_rate` | `2e-5` | Learning rate |
| `batch_size` | `2` | Batch size |
| `num_epochs` | `2` | Number of epochs |
| `temperature` | `0.02` | InfoNCE temperature |
| `train_size` | `25000` | Number of training examples |
| `eval_size` | `200` | Number of eval examples |
| `device` | `cuda` | Device (cuda/cpu) |
| `use_wandb` | `true` | Enable Weights & Biases logging |
| `--no-wandb` | - | Disable wandb (flag) |

## 📁 Data Format

Your soft-labeled JSON file should contain:
```json
[
  {
    "query": "what is the capital of france",
    "answers": ["Paris"],
    "passages": {
      "passage_text": ["Paris is the capital...", "France has many cities..."],
      "is_selected": [1, 0]  // Hard labels
      // OR "soft_labels": [0.1, 0.9]  // Soft labels
    }
  }
]
```

## 🚀 Automated Setup Script

Use the provided shell script to automate everything:
```bash
./run_training.sh YOUR_GOOGLE_DRIVE_FILE_ID
```

## 🔍 Examples

### Quick Standard Training
```bash
python -m src.bitter_retrieval.train --method standard_infonce --no-wandb
```

### Experiment with Different Methods
```bash
# Standard InfoNCE
python -m src.bitter_retrieval.train --method standard_infonce --run-name "standard_exp"

# Converted InfoNCE  
python -m src.bitter_retrieval.train --method converted_infonce --run-name "converted_exp"

# KL Soft InfoNCE
python -m src.bitter_retrieval.train --method kl_soft_infonce --run-name "kl_exp"
```

### Memory-Constrained Training
```bash
python -m src.bitter_retrieval.train \
    --method standard_infonce \
    --batch-size 1 \
    --train-size 5000 \
    --device cpu
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` to 1
- Reduce `train_size` 
- Use `--device cpu`

### Data Not Found
- Check that `data/msmarco/soft_labels_msmarco_3B_nopad_5000.json` exists
- Re-run `python download_data.py YOUR_FILE_ID`

### Import Errors
- Make sure you're in the repo root directory
- Run `./setup.sh` on fresh servers or `poetry install` if you already have Poetry
- Ensure you've sourced your bashrc: `source ~/.bashrc`

## 📊 Monitoring

By default, training metrics are logged to Weights & Biases. To disable:
```bash
python -m src.bitter_retrieval.train --method standard_infonce --no-wandb
```

## 🎉 That's It!

The simplified codebase makes it easy to get started. For more advanced usage, check the individual module docstrings in `src/bitter_retrieval/`. 
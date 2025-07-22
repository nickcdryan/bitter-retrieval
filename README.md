# Bitter Retrieval

A simplified training pipeline for retrieval models with LLM-guided signals.

## 🚀 Quick Start

**For fresh Linux servers:**
```bash
git clone <your-repo-url>
cd bitter-retrieval
./setup.sh
poetry run python setup_env.py  # Configure API keys
```

**Then run training:**
```bash
./run_training.sh <your_google_drive_file_id>
```

## 📖 Documentation

See [`QUICKSTART.md`](QUICKSTART.md) for detailed setup instructions and usage examples.

## 🔧 Features

- ✅ Automated Linux server setup
- ✅ Poetry-based dependency management with CUDA support
- ✅ PyTorch with CUDA 12.8 optimized for ML training
- ✅ Multiple training methods (InfoNCE variants)
- ✅ Google Drive data integration
- ✅ Weights & Biases monitoring
- ✅ Reproducible environments across different Linux distros

## 🏗️ Supported Systems

- Ubuntu/Debian
- Fedora/RHEL/CentOS
- Arch/Manjaro
- Python 3.8+

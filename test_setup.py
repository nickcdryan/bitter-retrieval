#!/usr/bin/env python3
"""
Test script to verify bitter-retrieval setup is working correctly.
"""

def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA devices: {torch.cuda.device_count()}")
            print(f"   Current device: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import datasets
        print(f"✅ Datasets: {datasets.__version__}")
    except ImportError as e:
        print(f"❌ Datasets import failed: {e}")
        return False
    
    try:
        import wandb
        print(f"✅ Wandb: {wandb.__version__}")
    except ImportError as e:
        print(f"❌ Wandb import failed: {e}")
        return False
    
    try:
        import gdown
        print(f"✅ Gdown: Available")
    except ImportError as e:
        print(f"❌ Gdown import failed: {e}")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy: {numpy.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    return True


def test_bitter_retrieval_imports():
    """Test that bitter-retrieval modules can be imported."""
    print("\n🔍 Testing bitter-retrieval modules...")
    
    try:
        from src.bitter_retrieval.config import load_config
        print("✅ Config module")
    except ImportError as e:
        print(f"❌ Config module failed: {e}")
        return False
    
    try:
        from src.bitter_retrieval.data import load_soft_labeled_data
        print("✅ Data module")
    except ImportError as e:
        print(f"❌ Data module failed: {e}")
        return False
    
    try:
        from src.bitter_retrieval.models import create_baseline_model
        print("✅ Models module")
    except ImportError as e:
        print(f"❌ Models module failed: {e}")
        return False
    
    try:
        from src.bitter_retrieval.utils import setup_logging
        print("✅ Utils module")
    except ImportError as e:
        print(f"❌ Utils module failed: {e}")
        return False
    
    try:
        from src.bitter_retrieval.auth import setup_authentication
        print("✅ Auth module")
    except ImportError as e:
        print(f"❌ Auth module failed: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality."""
    print("\n🔍 Testing basic functionality...")
    
    try:
        from src.bitter_retrieval.config import load_config
        config = load_config()
        print("✅ Configuration loading")
        print(f"   Default method: {config['method']}")
        print(f"   Device: {config['device']}")
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    try:
        import torch
        if torch.cuda.is_available():
            # Test CUDA memory allocation
            x = torch.randn(10, 10).cuda()
            y = torch.randn(10, 10).cuda()
            z = x @ y
            print("✅ CUDA tensor operations")
        else:
            print("⚠️  CUDA not available, skipping CUDA tests")
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("🚀 Bitter Retrieval Setup Verification")
    print("=" * 40)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test bitter-retrieval modules
    if not test_bitter_retrieval_imports():
        all_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! Your setup is ready for training.")
        print("\n💡 Next steps:")
        print("1. Get your Google Drive file ID for soft-labeled data")
        print("2. Run: ./run_training.sh <your_google_drive_file_id>")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 
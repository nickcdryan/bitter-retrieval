"""
Simple configuration for bitter-retrieval.
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Dict, Any


# Default configuration
DEFAULT_CONFIG = {
    # Data
    "soft_labels_path": "data/msmarco/soft_labels_msmarco_3B_nopad_5000.json",
    "train_size": 25000,
    "eval_size": 200,
    "eval_start_idx": 4000,
    
    # SQuAD
    "squad_num_titles": 100,
    "squad_questions_per_title": 5,
    "squad_num_examples": 100,
    "squad_test_size": 400,
    
    # Models
    "encoder_model": "nomic-ai/nomic-embed-text-v1-unsupervised", 
    "llm_model": "meta-llama/Llama-3.2-3B",
    "max_seq_length": 512,
    "generation_max_tokens": 40,
    
    # Training
    "learning_rate": 2e-5,
    "batch_size": 2,
    "grad_accumulation_steps": 8,
    "num_epochs": 2,
    "warmup_steps": 400,
    "eval_steps": 750,
    "temperature": 0.02,
    "teacher_temperature": 0.01,
    "student_temperature": 0.01,
    "margin": 1.0,
    "alpha": 1.0,
    "method": "standard_infonce",
    
    # System
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "use_wandb": True,
    "wandb_project": "bitter-retrieval",
    "run_name": None,
}


def load_config(config_path: str = None, **overrides) -> Dict[str, Any]:
    """
    Load configuration from file or defaults, with optional overrides.
    
    Args:
        config_path: Path to JSON config file (optional)
        **overrides: Keyword arguments to override config values
        
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    # Load from file if provided
    if config_path:
        with open(config_path, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)
    
    # Apply overrides
    config.update(overrides)
    
    # Ensure device is valid
    if config["device"] == "cuda" and not torch.cuda.is_available():
        config["device"] = "cpu"
        print("Warning: CUDA not available, using CPU")
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def parse_args() -> Dict[str, Any]:
    """Parse command line arguments and return config overrides."""
    parser = argparse.ArgumentParser(description="Train retrieval model")
    
    # Essential arguments only
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--method", type=str, 
                       choices=["standard_infonce", "converted_infonce", "kl_soft_infonce"],
                       help="Training method")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--num-epochs", type=int, help="Number of epochs")
    parser.add_argument("--temperature", type=float, help="Temperature")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--run-name", type=str, help="Run name")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    # Convert to config overrides
    overrides = {}
    if args.method:
        overrides["method"] = args.method
    if args.learning_rate:
        overrides["learning_rate"] = args.learning_rate
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.num_epochs:
        overrides["num_epochs"] = args.num_epochs
    if args.temperature:
        overrides["temperature"] = args.temperature
    if args.no_wandb:
        overrides["use_wandb"] = False
    if args.run_name:
        overrides["run_name"] = args.run_name
    if args.device:
        overrides["device"] = args.device
    if args.seed:
        overrides["seed"] = args.seed
    
    return load_config(args.config, **overrides) 
"""
Simple configuration for bitter-retrieval.
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Dict, Any


class ModelConfig:
    """Model-related configuration with attribute access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.encoder_model = config_dict.get("encoder_model", "nomic-ai/nomic-embed-text-v1-unsupervised")
        self.llm_model = config_dict.get("llm_model", "meta-llama/Llama-3.2-3B")
        self.max_seq_length = config_dict.get("max_seq_length", 512)
        self.generation_max_tokens = config_dict.get("generation_max_tokens", 40)


class Config:
    """Configuration class with attribute access."""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        if config_dict is None:
            config_dict = DEFAULT_CONFIG.copy()
        
        # Data
        self.soft_labels_path = config_dict.get("soft_labels_path", "data/msmarco/soft_labels_msmarco_3B_nopad_5000.json")
        self.train_size = config_dict.get("train_size", 25000)
        self.eval_size = config_dict.get("eval_size", 200)
        self.eval_start_idx = config_dict.get("eval_start_idx", 4000)
        
        # SQuAD
        self.squad_num_titles = config_dict.get("squad_num_titles", 100)
        self.squad_questions_per_title = config_dict.get("squad_questions_per_title", 5)
        self.squad_num_examples = config_dict.get("squad_num_examples", 100)
        self.squad_test_size = config_dict.get("squad_test_size", 400)
        
        # Model config
        self.model = ModelConfig(config_dict)
        
        # Training
        self.learning_rate = config_dict.get("learning_rate", 2e-5)
        self.batch_size = config_dict.get("batch_size", 2)
        self.grad_accumulation_steps = config_dict.get("grad_accumulation_steps", 8)
        self.num_epochs = config_dict.get("num_epochs", 2)
        self.warmup_steps = config_dict.get("warmup_steps", 400)
        self.eval_steps = config_dict.get("eval_steps", 750)
        self.temperature = config_dict.get("temperature", 0.02)
        self.teacher_temperature = config_dict.get("teacher_temperature", 0.01)
        self.student_temperature = config_dict.get("student_temperature", 0.01)
        self.margin = config_dict.get("margin", 1.0)
        self.alpha = config_dict.get("alpha", 1.0)
        self.method = config_dict.get("method", "standard_infonce")
        
        # System
        self.device = config_dict.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.seed = config_dict.get("seed", 42)
        self.use_wandb = config_dict.get("use_wandb", True)
        self.wandb_project = config_dict.get("wandb_project", "bitter-retrieval")
        self.run_name = config_dict.get("run_name", None)
        
        # Store original dict for compatibility
        self._config_dict = config_dict


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
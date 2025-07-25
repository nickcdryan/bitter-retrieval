"""
Configuration management for bitter retrieval training
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def get_default_config() -> Dict[str, Any]:
    """Get default training configuration"""
    return {
        # LOGGING
        "wandb_project": "bitter-retrieval",
        "run_name": "default-run",

        # MODELS
        # Decoder LLM for evaluation
        "llm_model": "Qwen/Qwen3-8B-Base", 
        # Embedding model trained for retrieval
        "encoder_model": "nomic-ai/nomic-embed-text-v1-unsupervised",
        
        # DATA PARAMS
        "dataset_name": "nickcdryan/ms_marco_softlabel_Qwen3-8B-Base_bf16",
        "num_data_examples": -1,  # Set to None or -1 to use all available training examples
        "encode_max_length": 512,  # BERT base max sequence length 512
        "llm_max_length": 1024,
        "generation_max_length": 900,
        "generation_max_tokens": 40,
        
        # TRAINING PARAMS
        "batch_size": 16,
        "learning_rate": 2e-5,
        "num_epochs": 2,
        "validation_frequency": 1000, # steps
        "gradient_clipping": True,  # Enable/disable gradient clipping
        "grad_clip_max_norm": 1.0,  # Maximum gradient norm
        "use_warmup": True,  # Enable/disable warmup
        "warmup_steps": 200,  # Number of warmup steps
        "use_lr_decay": False,  # Enable/disable linear decay after warmup

        # TRAINING METHOD AND PARAMS
        "training_method": "modular",  # "standard_infonce", "converted_infonce", "kl_soft_infonce", "modular"

        # MODULAR TRAINING METHOD - LOSS COMPONENT WEIGHTS
        "loss_components": {"kl": 0.5, "margin": 0.5},  # Default: KL + Margin

        # LOSS FUNCTION HPARAMS
        "infonce_temperature": 0.02, # for standard and converted
        "teacher_temp": .01, # for soft kl and modular
        "student_temp": .01, # for soft kl and modular
        "margin": 3.0, # for margin loss
        
        # EVALUATION DATASETS
        "squad_num_titles": 150,     # number of unique articles
        "squad_questions_per_title": 5, # how many questions associated with each article
        "squad_eval_examples": 100, # Small validation sets
        "squad_test_examples": 500,
        "msmarco_val_examples": 100,
        "msmarco_test_examples": 500,
        
        # MODEL SAVING
        "save_model": True,
        "model_save_path": "models/",
        
        # API TOKENS (from environment)
        "hf_token": os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
        "wandb_key": os.getenv("WANDB_API_KEY"),
        "gemini_key": os.getenv("GEMINI_API_KEY"),
    }


def flatten_dict(nested_dict: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flatten a nested dictionary for backwards compatibility"""
    items = []
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(flat_dict: Dict[str, Any], sep: str = '_') -> Dict[str, Any]:
    """Convert flat dictionary back to nested structure"""
    nested = {}
    for key, value in flat_dict.items():
        parts = key.split(sep)
        d = nested
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return nested


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Handle base_config inheritance
    if 'base_config' in yaml_config:
        base_path = config_path.parent / yaml_config['base_config']
        base_config = load_yaml_config(str(base_path))
        
        # Merge base config with current config (current overrides base)
        merged_config = merge_configs(base_config, yaml_config)
        # Remove base_config key from final config
        merged_config.pop('base_config', None)
        return merged_config
    
    return yaml_config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two configuration dictionaries"""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def yaml_to_flat_config(yaml_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert YAML config structure to flat config for backwards compatibility"""
    flat_config = {}
    
    # Map YAML structure to flat keys
    if 'logging' in yaml_config:
        flat_config.update({
            'wandb_project': yaml_config['logging'].get('wandb_project'),
            'run_name': yaml_config['logging'].get('run_name'),
        })
    
    if 'models' in yaml_config:
        flat_config.update({
            'llm_model': yaml_config['models'].get('llm_model'),
            'encoder_model': yaml_config['models'].get('encoder_model'),
        })
    
    if 'data' in yaml_config:
        flat_config.update({
            'dataset_name': yaml_config['data'].get('dataset_name'),
            'num_data_examples': yaml_config['data'].get('num_data_examples'),
            'encode_max_length': yaml_config['data'].get('encode_max_length'),
            'llm_max_length': yaml_config['data'].get('llm_max_length'),
            'generation_max_length': yaml_config['data'].get('generation_max_length'),
            'generation_max_tokens': yaml_config['data'].get('generation_max_tokens'),
        })
    
    if 'training' in yaml_config:
        flat_config.update({
            'batch_size': yaml_config['training'].get('batch_size'),
            'learning_rate': yaml_config['training'].get('learning_rate'),
            'num_epochs': yaml_config['training'].get('num_epochs'),
            'validation_frequency': yaml_config['training'].get('validation_frequency'),
            'gradient_clipping': yaml_config['training'].get('gradient_clipping'),
            'grad_clip_max_norm': yaml_config['training'].get('grad_clip_max_norm'),
            'use_warmup': yaml_config['training'].get('use_warmup'),
            'warmup_steps': yaml_config['training'].get('warmup_steps'),
            'use_lr_decay': yaml_config['training'].get('use_lr_decay'),
        })
    
    if 'method' in yaml_config:
        flat_config.update({
            'training_method': yaml_config['method'].get('training_method'),
            'loss_components': yaml_config['method'].get('loss_components'),
        })
    
    if 'loss_params' in yaml_config:
        flat_config.update({
            'infonce_temperature': yaml_config['loss_params'].get('infonce_temperature'),
            'teacher_temp': yaml_config['loss_params'].get('teacher_temp'),
            'student_temp': yaml_config['loss_params'].get('student_temp'),
            'margin': yaml_config['loss_params'].get('margin'),
        })
    
    if 'evaluation' in yaml_config:
        flat_config.update({
            'squad_num_titles': yaml_config['evaluation'].get('squad_num_titles'),
            'squad_questions_per_title': yaml_config['evaluation'].get('squad_questions_per_title'),
            'squad_eval_examples': yaml_config['evaluation'].get('squad_eval_examples'),
            'squad_test_examples': yaml_config['evaluation'].get('squad_test_examples'),
            'msmarco_val_examples': yaml_config['evaluation'].get('msmarco_val_examples'),
            'msmarco_test_examples': yaml_config['evaluation'].get('msmarco_test_examples'),
        })
    
    if 'saving' in yaml_config:
        flat_config.update({
            'save_model': yaml_config['saving'].get('save_model'),
            'model_save_path': yaml_config['saving'].get('model_save_path'),
        })
    
    # Remove None values
    flat_config = {k: v for k, v in flat_config.items() if v is not None}
    
    return flat_config


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load YAML config and convert to flat format for backwards compatibility"""
    yaml_config = load_yaml_config(config_path)
    flat_config = yaml_to_flat_config(yaml_config)
    
    # Start with default config and override with YAML values
    config = get_default_config()
    config.update(flat_config)
    
    # Add environment variables
    config.update({
        "hf_token": os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
        "wandb_key": os.getenv("WANDB_API_KEY"),
        "gemini_key": os.getenv("GEMINI_API_KEY"),
    })
    
    return config


def update_config(base_config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration with new values"""
    config = base_config.copy()
    config.update(updates)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration values"""
    # Check required fields
    required_fields = ["encoder_model", "llm_model", "training_method"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    # Validate training method
    valid_methods = ["standard_infonce", "converted_infonce", "kl_soft_infonce", "modular"]
    if config["training_method"] not in valid_methods:
        raise ValueError(f"Invalid training method: {config['training_method']}. Must be one of {valid_methods}")
    
    # Validate loss components for modular training
    if config["training_method"] == "modular":
        if "loss_components" not in config:
            raise ValueError("Modular training requires 'loss_components' in config")
        
        valid_losses = ["kl", "mse", "margin", "standard_infonce", "converted_infonce"]
        for loss_name in config["loss_components"]:
            if loss_name not in valid_losses:
                raise ValueError(f"Invalid loss component: {loss_name}. Must be one of {valid_losses}")
    
    # Validate margin parameter
    if any("margin" in str(config.get("loss_components", {})) for _ in [1]):
        if "margin" not in config or config["margin"] <= 0:
            raise ValueError("Margin loss requires positive 'margin' parameter")


def get_experiment_config(experiment_name: str) -> Dict[str, Any]:
    """Get configuration for specific experiments"""
    base_config = get_default_config()
    
    experiments = {
        "kl_only": {
            "training_method": "modular",
            "loss_components": {"kl": 1.0},
            "run_name": "kl-only"
        },
        "margin_only": {
            "training_method": "modular", 
            "loss_components": {"margin": 1.0},
            "run_name": "margin-only"
        },
        "kl_margin": {
            "training_method": "modular",
            "loss_components": {"kl": 0.5, "margin": 0.5},
            "run_name": "kl-margin"
        },
        "infonce_standard": {
            "training_method": "standard_infonce",
            "run_name": "infonce-standard"
        },
        "infonce_converted": {
            "training_method": "converted_infonce", 
            "run_name": "infonce-converted"
        },
        "kl_soft": {
            "training_method": "kl_soft_infonce",
            "run_name": "kl-soft"
        }
    }
    
    if experiment_name not in experiments:
        raise ValueError(f"Unknown experiment: {experiment_name}. Available: {list(experiments.keys())}")
    
    return update_config(base_config, experiments[experiment_name]) 
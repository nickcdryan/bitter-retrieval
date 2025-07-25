"""
Logging and wandb utilities
"""

import wandb
from typing import Dict, Any


def setup_wandb(config: Dict[str, Any]) -> None:
    """Setup wandb logging"""
    if config["wandb_key"]:
        print(f"Logging into wandb with key: {config['wandb_key'][:8]}...")
        wandb.login(key=config["wandb_key"])
    else:
        print("No wandb API key found - skipping automatic login")
    
    wandb.init(
        project=config["wandb_project"],
        name=config["run_name"],
        config=config
    )


def log_artifact(filepath: str, name: str = "training_script", type: str = "code") -> None:
    """Log training script as artifact for reproducibility"""
    artifact = wandb.Artifact(name=name, type=type)
    artifact.add_file(filepath)
    wandb.log_artifact(artifact) 
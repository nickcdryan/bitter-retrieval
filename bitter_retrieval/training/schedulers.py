"""
Learning rate scheduling utilities
"""

import torch
from typing import Dict, Any


def create_lr_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any], total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Create learning rate scheduler based on config"""
    
    def lr_lambda(step):
        if config["use_warmup"] and step < config["warmup_steps"]:
            return step / config["warmup_steps"]
        elif config["use_lr_decay"] and config["use_warmup"]:
            # Linear decay after warmup
            decay_steps = total_steps - config["warmup_steps"]
            decay_progress = (step - config["warmup_steps"]) / decay_steps
            return max(0.0, 1.0 - decay_progress)
        elif config["use_lr_decay"] and not config["use_warmup"]:
            # Linear decay from start
            decay_progress = step / total_steps
            return max(0.0, 1.0 - decay_progress)
        else:
            return 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) 
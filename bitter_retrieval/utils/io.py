"""
File I/O utilities
"""

import os
from typing import Dict, Any


def save_model(model: Any, config: Dict[str, Any]) -> None:
    """Save trained model"""
    if not config["save_model"]:
        return
    
    save_path = config["model_save_path"] + config["run_name"]
    print(f"Saving model to {save_path}")
    
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"Model saved successfully") 
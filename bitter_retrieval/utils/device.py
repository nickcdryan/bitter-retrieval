"""
Device and model setup utilities
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from typing import Dict, Any, Tuple


def setup_device_and_models(config: Dict[str, Any]) -> Tuple[torch.device, Any, Any, Any]:
    """Initialize device and load models"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Login to HF if token provided
    if config["hf_token"]:
        from huggingface_hub import login
        login(token=config["hf_token"])
    
    bert_tokenizer = AutoTokenizer.from_pretrained(config["encoder_model"], trust_remote_code=True)
    
    llm = AutoModelForCausalLM.from_pretrained(config["llm_model"]).eval()  # Keep on CPU
    llm_tokenizer = AutoTokenizer.from_pretrained(config["llm_model"])
    
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    
    return device, bert_tokenizer, llm, llm_tokenizer


def setup_encoder_model(config: Dict[str, Any], device: torch.device) -> Any:
    """Setup encoder model for training"""
    model = AutoModel.from_pretrained(config["encoder_model"], trust_remote_code=True).to(device)
    return model 
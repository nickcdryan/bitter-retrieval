"""
Text encoding utilities
"""

import torch
import torch.nn.functional as F
from typing import List, Any


def encode_texts(texts: List[str], model: Any, tokenizer: Any, max_length: int = 512, device: torch.device = None) -> torch.Tensor:
    """Encode texts with BERT-style model"""
    if device is None:
        device = next(model.parameters()).device
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    outputs = model(**inputs)
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
    sum_embeddings = masked_embeddings.sum(dim=1)
    count_tokens = attention_mask.sum(dim=1, keepdim=True)
    embeddings = sum_embeddings / count_tokens
    return F.normalize(embeddings, dim=-1) 
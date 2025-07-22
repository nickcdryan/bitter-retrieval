"""
Utility functions for bitter-retrieval.
"""

import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def encode_texts(
    texts: List[str], 
    model: AutoModel, 
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Encode texts using the provided model and tokenizer.
    Returns normalized embeddings tensor.
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Tokenize texts
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_length
    ).to(device)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling with attention mask
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    
    # Apply attention mask and compute mean
    masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
    sum_embeddings = masked_embeddings.sum(dim=1)
    count_tokens = attention_mask.sum(dim=1, keepdim=True)
    embeddings = sum_embeddings / count_tokens
    
    # L2 normalize embeddings
    return F.normalize(embeddings, dim=-1)


def compute_f1_score(prediction: str, reference: str) -> float:
    """Compute token-level F1 score between prediction and reference."""
    from collections import Counter
    
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    overlap = pred_counter & ref_counter
    overlap_count = sum(overlap.values())

    precision = overlap_count / len(pred_tokens)
    recall = overlap_count / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, reference: str) -> float:
    """Compute exact match score between prediction and reference."""
    return float(prediction.lower().strip() == reference.lower().strip())


def generate_answer(
    query: str, 
    context: str, 
    llm_model, 
    llm_tokenizer,
    max_tokens: int = 40,
    device: Optional[torch.device] = None
) -> str:
    """Generate answer using LLM for the given query and context."""
    if device is None:
        device = next(llm_model.parameters()).device
    
    prompt = f"Question: {query} Context: {context} Answer:"
    inputs = llm_tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=900
    ).to(device)
    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = llm_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=llm_tokenizer.eos_token_id,
            eos_token_id=llm_tokenizer.eos_token_id
        )

    generated_tokens = outputs[0, prompt_length:]
    generated_text = llm_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text.strip()


def compute_llm_loss(
    query: str,
    context: str, 
    answer: str,
    llm_model,
    llm_tokenizer,
    device: Optional[torch.device] = None
) -> float:
    """Compute LLM loss for answer generation given query and context."""
    if device is None:
        device = next(llm_model.parameters()).device
    
    input_text = f"Question: {query} Context: {context} Answer: {answer}"
    inputs = llm_tokenizer(
        input_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=1024
    ).to(device)
    
    with torch.no_grad():
        outputs = llm_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
    
    return loss 
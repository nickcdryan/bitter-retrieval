"""
Model training functions for bitter-retrieval.
"""

import logging
from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from .data import should_skip_item, convert_soft_to_hard, load_squad_data
from .utils import encode_texts
from .evaluation import SquadEvaluator

logger = logging.getLogger(__name__)


def precompute_squad_embeddings(corpus_texts: List[str], model: AutoModel, tokenizer: AutoTokenizer) -> torch.Tensor:
    """Precompute embeddings for SQuAD corpus."""
    model.eval()
    embeddings = []
    logger.info("Precomputing SQuAD embeddings...")
    
    with torch.no_grad():
        for text in tqdm(corpus_texts, desc="Encoding SQuAD"):
            emb = encode_texts([f"passage: {text}"], model, tokenizer)
            embeddings.append(emb.cpu())
    
    return torch.cat(embeddings, dim=0).cuda()


def run_squad_validation(model: AutoModel, tokenizer: AutoTokenizer, config: Dict[str, Any], wandb_logger, train_step: int):
    """Run SQuAD validation and log results."""
    try:
        # Load SQuAD data if not already loaded (simplified approach)  
        squad_qa_data, squad_corpus, _ = load_squad_data(config)
        
        # Precompute embeddings
        squad_embeddings = precompute_squad_embeddings(squad_corpus, model, tokenizer)
        
        # Simplified evaluation without full evaluator class
        num_eval_examples = min(100, len(squad_qa_data))
        squad_results = evaluate_squad_simple(
            model, tokenizer, squad_qa_data[:num_eval_examples], squad_embeddings, squad_corpus
        )
        
        logger.info(f"SQuAD validation results: {squad_results}")
        
        # Log to wandb
        if wandb_logger:
            val_metrics = {
                "SQuAD F1": squad_results['F1_Score'],
                "SQuAD EM": squad_results['Exact_Match'],
                "SQuAD Retrieval": squad_results['Retrieval_Accuracy'],
                "SQuAD Loss": squad_results['LLM_Loss'],
            }
            wandb_logger.log(val_metrics, step=train_step)
        
        model.train()  # Switch back to training mode
        return squad_results
        
    except Exception as e:
        logger.warning(f"SQuAD validation failed: {e}")
        model.train()
        return None


def evaluate_squad_simple(model: AutoModel, tokenizer: AutoTokenizer, qa_data: List[Dict], 
                         corpus_embeddings: torch.Tensor, corpus_texts: List[str]) -> Dict[str, float]:
    """Simplified SQuAD evaluation for validation during training."""
    model.eval()
    retrieval_hits = []
    
    with torch.no_grad():
        for item in tqdm(qa_data, desc="SQuAD validation"):
            if item.get("answer") == "Unanswerable":
                continue

            question = item["question"]
            correct_context_idx = item["context_idx"]

            # Encode question and retrieve top context
            question_emb = encode_texts([f"query: {question}"], model, tokenizer)
            similarities = torch.matmul(question_emb, corpus_embeddings.T).squeeze(0)
            best_context_idx = similarities.argmax().item()

            # Check if we retrieved the correct context
            retrieval_hit = (best_context_idx == correct_context_idx)
            retrieval_hits.append(retrieval_hit)

    return {
        "Retrieval_Accuracy": float(torch.tensor(retrieval_hits).float().mean()),
        "LLM_Loss": 0.0,  # Simplified - skip LLM loss computation for speed
        "Exact_Match": 0.0,  # Simplified - skip generation for speed
        "F1_Score": 0.0,  # Simplified - skip generation for speed
        "Num_Examples": len(retrieval_hits)
    }


def create_model_and_optimizer(config: Dict[str, Any]):
    """Create model, optimizer, and scheduler."""
    device = torch.device(config["device"])
    
    model = AutoModel.from_pretrained(
        config["encoder_model"], 
        trust_remote_code=True
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"]
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=10000  # Estimate
    )
    
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    return model, optimizer, scheduler


def train_standard_infonce(
    config: Dict[str, Any],
    train_data: List[Dict[str, Any]],
    encoder_tokenizer: AutoTokenizer,
    evaluator=None,
    wandb_logger=None
) -> AutoModel:
    """Train model using standard InfoNCE with original hard labels."""
    logger.info("Training Standard InfoNCE on original labels")
    
    model, optimizer, scheduler = create_model_and_optimizer(config)
    device = torch.device(config["device"])
    train_step = 0
    
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        num_batches = 0

        for i in tqdm(range(0, len(train_data), config["batch_size"]), 
                     desc=f"Epoch {epoch+1}"):
            batch_items = train_data[i:i+config["batch_size"]]
            batch_loss = 0
            valid_items = 0

            for item in batch_items:
                if should_skip_item(item):
                    continue

                query = item['query']
                passages = item['passages']['passage_text']
                hard_labels = item['passages']['is_selected']

                # Find positive and negative examples
                pos_idx = hard_labels.index(1) if 1 in hard_labels else 0
                neg_indices = [j for j, label in enumerate(hard_labels) if label == 0]
                if not neg_indices:
                    continue

                # Encode texts
                query_emb = encode_texts([f"query: {query}"], model, encoder_tokenizer, device=device)
                pos_emb = encode_texts([f"passage: {passages[pos_idx]}"], model, encoder_tokenizer, device=device)
                neg_embs = encode_texts([f"passage: {passages[i]}" for i in neg_indices], model, encoder_tokenizer, device=device)

                # Compute similarities
                pos_sim = torch.sum(query_emb * pos_emb, dim=1)
                neg_sims = torch.sum(query_emb.unsqueeze(1) * neg_embs, dim=2).squeeze(0)

                # InfoNCE loss with temperature
                logits = torch.cat([pos_sim, neg_sims]) / config["temperature"]
                labels = torch.zeros(1, dtype=torch.long).to(device)
                loss = F.cross_entropy(logits.unsqueeze(0), labels)

                batch_loss += loss
                valid_items += 1

            if valid_items > 0:
                batch_loss = batch_loss / valid_items
                
                # Log metrics
                if wandb_logger:
                    wandb_logger.log({"Contrastive loss": batch_loss.item()}, step=train_step)
                
                logger.info(f"Step {train_step}: train_loss={batch_loss.item():.4f}")

                # Periodic validation
                if train_step % config.get("eval_steps", 750) == 0:
                    run_squad_validation(model, encoder_tokenizer, config, wandb_logger, train_step)

                # Backward pass
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += batch_loss.item()
                num_batches += 1

            train_step += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    return model


def train_converted_infonce(
    config: Dict[str, Any],
    train_data: List[Dict[str, Any]],
    encoder_tokenizer: AutoTokenizer,
    evaluator=None,
    wandb_logger=None
) -> AutoModel:
    """Train model using InfoNCE with converted soft-to-hard labels."""
    logger.info("Training Converted InfoNCE on LLM-converted labels")
    
    model, optimizer, scheduler = create_model_and_optimizer(config)
    device = torch.device(config["device"])
    train_step = 0
    
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        num_batches = 0

        for i in tqdm(range(0, len(train_data), config["batch_size"]), 
                     desc=f"Epoch {epoch+1}"):
            batch_items = train_data[i:i+config["batch_size"]]
            batch_loss = 0
            valid_items = 0

            for item in batch_items:
                if should_skip_item(item):
                    continue

                query = item['query']
                passages = item['passages']['passage_text']
                soft_labels = item['passages']['soft_labels']
                
                # Convert soft labels to hard labels
                hard_labels = convert_soft_to_hard(soft_labels)

                # Find positive and negative examples
                pos_idx = hard_labels.index(1) if 1 in hard_labels else 0
                neg_indices = [j for j, label in enumerate(hard_labels) if label == 0]
                if not neg_indices:
                    continue

                # Encode texts
                query_emb = encode_texts([f"query: {query}"], model, encoder_tokenizer, device=device)
                pos_emb = encode_texts([f"passage: {passages[pos_idx]}"], model, encoder_tokenizer, device=device)
                neg_embs = encode_texts([f"passage: {passages[i]}" for i in neg_indices], model, encoder_tokenizer, device=device)

                # Compute similarities
                pos_sim = torch.sum(query_emb * pos_emb, dim=1)
                neg_sims = torch.sum(query_emb.unsqueeze(1) * neg_embs, dim=2).squeeze(0)

                # InfoNCE loss with temperature
                logits = torch.cat([pos_sim, neg_sims]) / config["temperature"]
                labels = torch.zeros(1, dtype=torch.long).to(device)
                loss = F.cross_entropy(logits.unsqueeze(0), labels)

                batch_loss += loss
                valid_items += 1

            if valid_items > 0:
                batch_loss = batch_loss / valid_items
                
                # Log metrics
                if wandb_logger:
                    wandb_logger.log({"Contrastive loss": batch_loss.item()}, step=train_step)
                
                logger.info(f"Step {train_step}: train_loss={batch_loss.item():.4f}")

                # Periodic validation
                if train_step % config.get("eval_steps", 750) == 0:
                    run_squad_validation(model, encoder_tokenizer, config, wandb_logger, train_step)

                # Backward pass
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += batch_loss.item()
                num_batches += 1

            train_step += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    return model


def train_kl_soft_infonce(
    config: Dict[str, Any],
    train_data: List[Dict[str, Any]],
    encoder_tokenizer: AutoTokenizer,
    evaluator=None,
    wandb_logger=None
) -> AutoModel:
    """Train model using KL divergence with soft LLM loss distributions."""
    logger.info("Training KL Soft InfoNCE with LLM loss distributions")
    
    model, optimizer, scheduler = create_model_and_optimizer(config)
    device = torch.device(config["device"])
    train_step = 0
    
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        num_batches = 0

        for i in tqdm(range(0, len(train_data), config["batch_size"]), 
                     desc=f"Epoch {epoch+1}"):
            batch_items = train_data[i:i+config["batch_size"]]
            batch_loss = 0
            valid_items = 0

            for item in batch_items:
                if should_skip_item(item):
                    continue

                query = item['query']
                passages = item['passages']['passage_text']
                soft_labels = item['passages']['soft_labels']

                if len(passages) < 2:
                    continue

                # Encode texts
                query_emb = encode_texts([f"query: {query}"], model, encoder_tokenizer, device=device)
                passage_embs = encode_texts([f"passage: {p}" for p in passages], model, encoder_tokenizer, device=device)

                # Compute similarities
                similarities = torch.sum(query_emb.unsqueeze(1) * passage_embs, dim=2).squeeze(0)

                # Convert to probabilities with student temperature
                student_logits = similarities / config["student_temperature"]
                student_probs = F.log_softmax(student_logits, dim=0)

                # Teacher probabilities from soft labels (negative loss -> probability)
                teacher_logits = torch.tensor([-loss for loss in soft_labels], device=device) / config["teacher_temperature"]
                teacher_probs = F.softmax(teacher_logits, dim=0)

                # KL divergence loss
                kl_loss = F.kl_div(student_probs, teacher_probs, reduction='sum')

                batch_loss += kl_loss
                valid_items += 1

            if valid_items > 0:
                batch_loss = batch_loss / valid_items
                
                # Log metrics
                if wandb_logger:
                    wandb_logger.log({"KL loss": batch_loss.item()}, step=train_step)
                
                logger.info(f"Step {train_step}: train_loss={batch_loss.item():.4f}")

                # Periodic validation
                if train_step % config.get("eval_steps", 750) == 0:
                    run_squad_validation(model, encoder_tokenizer, config, wandb_logger, train_step)

                # Backward pass
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += batch_loss.item()
                num_batches += 1

            train_step += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    return model


def train_model(
    config: Dict[str, Any],
    train_data: List[Dict[str, Any]],
    encoder_tokenizer: AutoTokenizer,
    evaluator=None,
    wandb_logger=None
) -> AutoModel:
    """Train model using the specified method."""
    method = config["method"]
    
    if method == "standard_infonce":
        return train_standard_infonce(config, train_data, encoder_tokenizer, evaluator, wandb_logger)
    elif method == "converted_infonce":
        return train_converted_infonce(config, train_data, encoder_tokenizer, evaluator, wandb_logger)
    elif method == "kl_soft_infonce":
        return train_kl_soft_infonce(config, train_data, encoder_tokenizer, evaluator, wandb_logger)
    else:
        raise ValueError(f"Unknown training method: {method}")


def create_baseline_model(config: Dict[str, Any]) -> AutoModel:
    """Create baseline encoder model without training."""
    logger.info(f"Creating baseline model: {config['encoder_model']}")
    model = AutoModel.from_pretrained(
        config["encoder_model"], 
        trust_remote_code=True
    ).to(torch.device(config["device"]))
    
    return model 
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


def run_squad_validation(model: AutoModel, tokenizer: AutoTokenizer, config: Dict[str, Any], wandb_logger, train_step: int, llm_model=None, llm_tokenizer=None):
    """Run SQuAD validation and log results."""
    try:
        # Load SQuAD data if not already loaded
        squad_qa_data, squad_corpus, _ = load_squad_data(config)
        
        # Precompute embeddings
        squad_embeddings = precompute_squad_embeddings(squad_corpus, model, tokenizer)
        
        # Full evaluation with all metrics
        num_eval_examples = min(100, len(squad_qa_data))
        squad_results = evaluate_squad_full(
            model, tokenizer, squad_qa_data[:num_eval_examples], squad_embeddings, squad_corpus, llm_model, llm_tokenizer
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


def generate_answer(query: str, context: str, llm_model, llm_tokenizer, max_tokens: int = 40, device=None) -> str:
    """Generate answer autoregressively with stopping criteria."""
    if device is None:
        device = next(llm_model.parameters()).device
        
    prompt = f"Question: {query} Context: {context} Answer:"
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900).to(device)
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


def compute_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 score."""
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
    """Compute exact match score."""
    return float(prediction.lower().strip() == reference.lower().strip())


def evaluate_squad_full(model: AutoModel, tokenizer: AutoTokenizer, qa_data: List[Dict], 
                       corpus_embeddings: torch.Tensor, corpus_texts: List[str], 
                       llm_model=None, llm_tokenizer=None) -> Dict[str, float]:
    """Full SQuAD evaluation with all metrics, matching the working code exactly."""
    model.eval()
    retrieval_hits = []  # Did we retrieve the correct context?
    llm_losses = []
    exact_matches = []
    f1_scores = []
    
    device = next(model.parameters()).device

    with torch.no_grad():
        for item in tqdm(qa_data, desc="SQuAD evaluation"):
            if item.get("answer") == "Unanswerable":
                continue

            question = item["question"]
            correct_answer = item["answer"]
            correct_context_idx = item["context_idx"]

            # Encode question and retrieve top context
            question_emb = encode_texts([f"query: {question}"], model, tokenizer)
            similarities = torch.matmul(question_emb, corpus_embeddings.T).squeeze(0)
            best_context_idx = similarities.argmax().item()
            best_context = corpus_texts[best_context_idx]

            # Check if we retrieved the correct context
            retrieval_hit = (best_context_idx == correct_context_idx)
            retrieval_hits.append(retrieval_hit)

            # Compute LLM loss with retrieved context (if LLM is available)
            if llm_model is not None and llm_tokenizer is not None:
                input_text = f"Question: {question} Context: {best_context} Answer: {correct_answer}"
                inputs = llm_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
                labels = inputs["input_ids"].clone()

                target_start_idx = input_text.rfind("Answer: ") + len("Answer: ")
                target_start_token_idx = llm_tokenizer(input_text[:target_start_idx], return_tensors="pt")["input_ids"].shape[1] - 1
                labels[:, :target_start_token_idx] = -100
                labels[labels == llm_tokenizer.pad_token_id] = -100

                outputs = llm_model(**inputs)
                logits = outputs.logits[0, :-1, :].contiguous()
                labels = labels[0, 1:].contiguous()
                valid_positions = labels != -100

                if valid_positions.sum() > 0:
                    loss = F.cross_entropy(logits[valid_positions], labels[valid_positions])
                    llm_losses.append(loss.item())

                # Generate answer and compute EM/F1
                try:
                    generated_answer = generate_answer(question, best_context, llm_model, llm_tokenizer, device=device)
                    em = compute_exact_match(generated_answer, correct_answer)
                    f1 = compute_f1(generated_answer, correct_answer)
                    exact_matches.append(em)
                    f1_scores.append(f1)
                except:
                    continue

    return {
        "Retrieval_Accuracy": float(torch.tensor(retrieval_hits).float().mean()) if retrieval_hits else 0.0,
        "LLM_Loss": float(torch.tensor(llm_losses).mean()) if llm_losses else 0.0,
        "Exact_Match": float(torch.tensor(exact_matches).mean()) if exact_matches else 0.0,
        "F1_Score": float(torch.tensor(f1_scores).mean()) if f1_scores else 0.0,
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
    wandb_logger=None,
    llm_model=None,
    llm_tokenizer=None
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
                    run_squad_validation(model, encoder_tokenizer, config, wandb_logger, train_step, llm_model, llm_tokenizer)

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
    wandb_logger=None,
    llm_model=None,
    llm_tokenizer=None
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
                    run_squad_validation(model, encoder_tokenizer, config, wandb_logger, train_step, llm_model, llm_tokenizer)

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
    wandb_logger=None,
    llm_model=None,
    llm_tokenizer=None
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
                    run_squad_validation(model, encoder_tokenizer, config, wandb_logger, train_step, llm_model, llm_tokenizer)

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
    wandb_logger=None,
    llm_model=None,
    llm_tokenizer=None
) -> AutoModel:
    """Train model using the specified method."""
    method = config["method"]
    
    if method == "standard_infonce":
        return train_standard_infonce(config, train_data, encoder_tokenizer, evaluator, wandb_logger, llm_model, llm_tokenizer)
    elif method == "converted_infonce":
        return train_converted_infonce(config, train_data, encoder_tokenizer, evaluator, wandb_logger, llm_model, llm_tokenizer)
    elif method == "kl_soft_infonce":
        return train_kl_soft_infonce(config, train_data, encoder_tokenizer, evaluator, wandb_logger, llm_model, llm_tokenizer)
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
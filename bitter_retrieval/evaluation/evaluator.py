"""
Main evaluation orchestrator
"""

import torch
import torch.nn.functional as F
import numpy as np
import asyncio
from tqdm import tqdm
from typing import Dict, Any, List, Tuple

from ..utils.encoding import encode_texts
from ..models.encoder import precompute_embeddings
from ..models.llm import generate_answers_batch
from .metrics import compute_f1, compute_exact_match
from .llm_judge import llm_judge_answer_async, batch_llm_judge


def evaluate_retrieval(model: Any, qa_data: List[Dict], corpus_embeddings: torch.Tensor, 
                      corpus_texts: List[str], llm: Any, llm_tokenizer: Any, 
                      bert_tokenizer: Any, config: Dict[str, Any], num_examples: int = 500) -> Dict[str, float]:
    """Optimized evaluation with batch encoding, batch generation, and async LLM judge"""
    model.eval()
    
    # Filter valid examples
    valid_items = [item for item in qa_data[:num_examples] if item["answer"] != "Unanswerable"]
    if not valid_items:
        return {"Retrieval_Accuracy": 0.0, "LLM_Loss": 0.0, "Exact_Match": 0.0, "F1_Score": 0.0, "LLM_Judge": 0.0, "Num_Examples": 0}
    
    print(f"Processing {len(valid_items)} valid examples")
    
    # OPTIMIZATION 1: Batch encode all questions at once
    questions = [f"query: {item['question']}" for item in valid_items]
    with torch.no_grad():
        question_embs = encode_texts(questions, model, bert_tokenizer, config["encode_max_length"])
    
    retrieval_hits = []
    llm_losses = []
    exact_matches = []
    f1_scores = []
    
    # Collect data for batch processing
    generation_pairs = []  # (question, best_context) pairs
    judge_data = []        # (question, correct_answer) pairs for LLM judge
    
    with torch.no_grad():
        for i, item in enumerate(tqdm(valid_items, desc="Processing retrieval and LLM loss")):
            question = item["question"]
            correct_answer = item["answer"]
            correct_context_idx = item["context_idx"]
            
            # Use pre-computed question embedding
            question_emb = question_embs[i:i+1]
            similarities = torch.matmul(question_emb, corpus_embeddings.T).squeeze(0)
            best_context_idx = similarities.argmax().item()
            best_context = corpus_texts[best_context_idx]
            
            retrieval_hit = (best_context_idx == correct_context_idx)
            retrieval_hits.append(retrieval_hit)
            
            # Compute LLM loss
            device = next(llm.parameters()).device
            input_text = f"Question: {question} Context: {best_context} Answer: {correct_answer}"
            inputs = llm_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=config["llm_max_length"]).to(device)
            labels = inputs["input_ids"].clone()
            
            target_start_idx = input_text.rfind("Answer: ") + len("Answer: ")
            target_start_token_idx = llm_tokenizer(input_text[:target_start_idx], return_tensors="pt")["input_ids"].shape[1] - 1
            labels[:, :target_start_token_idx] = -100
            labels[labels==llm_tokenizer.pad_token_id] = -100
            
            outputs = llm(**inputs)
            logits = outputs.logits[0, :-1, :].contiguous()
            labels = labels[0, 1:].contiguous()
            valid_positions = labels != -100
            
            if valid_positions.sum() > 0:
                loss = F.cross_entropy(logits[valid_positions], labels[valid_positions])
                llm_losses.append(loss.item())
            
            # Collect for batch processing
            generation_pairs.append((question, best_context))
            judge_data.append((question, correct_answer))
    
    # OPTIMIZATION 2: Batch answer generation
    print("Generating answers in batches...")
    try:
        generated_answers = generate_answers_batch(
            generation_pairs, llm, llm_tokenizer, 
            config["generation_max_length"], config["generation_max_tokens"], batch_size=8
        )
        
        # Compute EM and F1 scores
        for i, (_, correct_answer) in enumerate(judge_data):
            if i < len(generated_answers):
                generated_answer = generated_answers[i]
                em = compute_exact_match(generated_answer, correct_answer)
                f1 = compute_f1(generated_answer, correct_answer)
                exact_matches.append(em)
                f1_scores.append(f1)
    except Exception as e:
        print(f"Batch generation failed: {e}")
        generated_answers = []
    
    # OPTIMIZATION 3: Async LLM judge
    llm_judge_scores = []
    if generated_answers and len(generated_answers) == len(judge_data):
        print("Running async LLM judge...")
        try:
            async def run_async_judge():
                judge_tasks = [
                    llm_judge_answer_async(question, correct_answer, generated_answer)
                    for (question, correct_answer), generated_answer in zip(judge_data, generated_answers)
                ]
                return await batch_llm_judge(judge_tasks)
            
            # Run async LLM judge
            llm_judge_scores = asyncio.run(run_async_judge())
        except Exception as e:
            print(f"Async LLM judge failed: {e}")
            llm_judge_scores = [0.0] * len(generated_answers)
    
    return {
        "Retrieval_Accuracy": np.mean(retrieval_hits) if retrieval_hits else 0.0,
        "LLM_Loss": np.mean(llm_losses) if llm_losses else 0.0,
        "Exact_Match": np.mean(exact_matches) if exact_matches else 0.0,
        "F1_Score": np.mean(f1_scores) if f1_scores else 0.0,
        "LLM_Judge": np.mean(llm_judge_scores) if llm_judge_scores else 0.0,
        "Num_Examples": len(retrieval_hits)
    }


def run_validation(model: Any, squad_qa_data: List[Dict], squad_corpus: List[str],
                  msmarco_qa_data: List[Dict], msmarco_corpus: List[str], 
                  llm: Any, llm_tokenizer: Any, bert_tokenizer: Any, config: Dict[str, Any], device: torch.device) -> Dict[str, Dict]:
    """Run both SQuAD and MS MARCO validation"""
    # Move LLM to GPU for validation
    llm.to(device)
    
    try:
        # SQuAD validation
        squad_embeddings = precompute_embeddings(squad_corpus, model, bert_tokenizer, config, "SQuAD")
        squad_results = evaluate_retrieval(
            model, squad_qa_data[:config["squad_eval_examples"]], squad_embeddings, 
            squad_corpus, llm, llm_tokenizer, bert_tokenizer, config
        )
        print("SQuAD validation:", squad_results)
        
        # MS MARCO validation  
        msmarco_embeddings = precompute_embeddings(msmarco_corpus, model, bert_tokenizer, config, "MS MARCO")
        msmarco_results = evaluate_retrieval(
            model, msmarco_qa_data, msmarco_embeddings, msmarco_corpus, 
            llm, llm_tokenizer, bert_tokenizer, config
        )
        print("MS MARCO validation:", msmarco_results)
        
        return {
            "squad": squad_results,
            "msmarco": msmarco_results
        }
    finally:
        # Move LLM back to CPU and clear cache
        llm.cpu()
        torch.cuda.empty_cache() 
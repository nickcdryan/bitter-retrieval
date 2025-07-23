#!/usr/bin/env python3

# poetry run python train_retrieval.py

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
import json
import random
import numpy as np
import os
from collections import defaultdict, Counter
import wandb
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("python-dotenv not installed - using system environment variables only")


def get_config():
    """Training configuration"""
    return {
        # Models
        "llm_model": "Qwen/Qwen3-8B-Base", 
        "encoder_model": "google-bert/bert-base-uncased", #"nomic-ai/nomic-embed-text-v1-unsupervised",
        
        # Max lengths
        "encode_max_length": 512,  # BERT base max sequence length 512
        "llm_max_length": 1024,
        "generation_max_length": 900,
        "generation_max_tokens": 40,
        
        # Training params
        "batch_size": 32,
        "learning_rate": 2e-5,
        "num_epochs": 2,  # Just 1 epoch for testing
        "warmup_steps": 200,  # Fewer warmup steps
        "validation_frequency": 500,  # Validate more frequently
        
        # Training method and params
        "training_method": "standard_infonce",  # "standard_infonce", "converted_infonce", "kl_soft_infonce"
        "temperature": 0.02,
        "teacher_temp": 0.1,
        "student_temp": 0.05,
        "margin": 1.0,
        
        # Data params
        "dataset_name": "nickcdryan/ms_marco_softlabel_Qwen3-8B-Base_bf16",
        "num_data_examples": -1,  # Set to None or -1 to use all available training examples

        # build our corpus, including every single passages belonging to each squad_num_titles to make retrieval harder
        # end up with squad_questions_per_title * squad_num_titles total questions, and corpus is about 45 * squad_num_titles passages
        "squad_num_titles": 150,     # number of unique articles (with a number of questions and contexts per article)
        "squad_questions_per_title": 5, # how many questions associated with each article
        "squad_eval_examples": 100, # Small validation sets
        "squad_test_examples": 500,
        "msmarco_val_examples": 100,
        "msmarco_test_examples": 500,


        # LITE!
        # build our corpus, including every single passages belonging to each squad_num_titles to make retrieval harder
        # end up with squad_questions_per_title * squad_num_titles total questions, and corpus is about 45 * squad_num_titles passages
        # "squad_num_titles": 10,     # number of unique articles (with a number of questions and contexts per article)
        # "squad_questions_per_title": 5, # how many questions associated with each article
        # "squad_eval_examples": 5, # Small validation sets
        # "squad_test_examples": 5,
        # "msmarco_val_examples": 5,
        # "msmarco_test_examples": 5,
        
        # Logging
        "wandb_project": "bitter-retrieval",
        "run_name": "standard_infonce-BERT-fulltraining-epoch:2-batch:32",
        
        # Model saving
        "save_model": True,
        "model_save_path": "models/",
        
        # Tokens (from environment)
        "hf_token": os.getenv("HF_TOKEN"),
        "wandb_key": os.getenv("WANDB_API_KEY"),
        "gemini_key": os.getenv("GEMINI_API_KEY"),
    }


def setup_device_and_models(config):
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


def preprocess_squad_title_diverse(data, num_titles=50, questions_per_title=5):
    """Take a few questions from each title but include ALL contexts for selected titles"""
    title_groups = defaultdict(list)
    for item in data:
        title = item.get("title", "Unknown")
        title_groups[title].append(item)

    print(f"Found {len(title_groups)} unique titles in dataset")
    
    selected_titles = random.sample(list(title_groups.keys()), min(num_titles, len(title_groups)))
    
    qa_data = []
    corpus_texts = []
    context_to_idx = {}
    title_to_contexts = defaultdict(set)
    
    for title in selected_titles:
        title_items = title_groups[title]
        
        title_contexts = set()
        for item in title_items:
            title_contexts.add(item["context"])
        
        for context in title_contexts:
            if context not in context_to_idx:
                context_to_idx[context] = len(corpus_texts)
                corpus_texts.append(context)
                title_to_contexts[title].add(len(corpus_texts) - 1)
        
        sampled_questions = random.sample(title_items, min(questions_per_title, len(title_items)))
        
        for item in sampled_questions:
            context = item["context"]
            question = item["question"]
            answers = item["answers"]
            
            if answers["text"]:
                answer = answers["text"][0]
            else:
                answer = "Unanswerable"
            
            qa_data.append({
                "question": question,
                "answer": answer,
                "context_idx": context_to_idx[context],
                "title": title,
                "correct_context": context,
                "title_context_indices": list(title_to_contexts[title])
            })
    
    contexts_per_title = [len(contexts) for contexts in title_to_contexts.values()]
    print(f"Selected {len(selected_titles)} titles")
    print(f"Total questions: {len(qa_data)}")
    print(f"Total unique contexts: {len(corpus_texts)}")
    print(f"Avg contexts per title: {sum(contexts_per_title) / len(contexts_per_title):.1f}")
    
    return qa_data, corpus_texts, title_to_contexts


def preprocess_msmarco_validation(config, num_examples=500):
    """
    Create validation set from MS MARCO validation split.
    Similar to SQuAD: questions/answers with larger corpus including distractors.
    """
    print("Loading MS MARCO validation dataset...")
    ms_marco_dataset = load_dataset(config["dataset_name"])
    validation_data = list(ms_marco_dataset["validation"])
    
    # Sample examples for validation
    sampled_data = random.sample(validation_data, min(num_examples, len(validation_data)))
    
    qa_data = []
    corpus_texts = []
    context_to_idx = {}
    
    # Add empty string context once (it appears in every example as last passage)
    empty_string = ""
    context_to_idx[empty_string] = len(corpus_texts)
    corpus_texts.append(empty_string)
    
    # Collect all unique passages from sampled examples
    for item in sampled_data:
        passages = item['passages']['passage_text']
        
        # Add all non-empty passages to corpus
        for passage in passages:
            if passage != empty_string and passage not in context_to_idx:
                context_to_idx[passage] = len(corpus_texts)
                corpus_texts.append(passage)
    
    # Create QA pairs
    for item in sampled_data:
        query = item['query']
        passages = item['passages']['passage_text']
        answers = item.get('answers', [])
        
        if not answers:
            continue
            
        # Find correct context based on original labels
        if 'is_selected' in item['passages']:
            hard_labels = item['passages']['is_selected']
            correct_idx = hard_labels.index(1) if 1 in hard_labels else 0
        else:
            # Convert soft labels to find correct passage
            soft_labels = item['passages']['soft_labels']
            correct_idx = soft_labels.index(min(soft_labels))
        
        correct_context = passages[correct_idx]
        
        qa_data.append({
            "question": query,
            "answer": answers[0],
            "context_idx": context_to_idx[correct_context],
            "correct_context": correct_context,
            "source": "msmarco"
        })
    
    print(f"MS MARCO validation: {len(qa_data)} questions, {len(corpus_texts)} unique contexts")
    print(f"Empty string context at index: {context_to_idx[empty_string]}")
    
    return qa_data, corpus_texts


def should_skip_item(item):
    """Check if training item should be skipped"""
    if not item['answers']:
        return True
    if len(item['passages']['passage_text']) < 2:
        return True
    
    if hasattr(item['passages'], 'is_selected'):
        hard_labels = item['passages']['is_selected']
    else:
        soft_labels = item['passages']['soft_labels']
        hard_labels = convert_soft_to_hard(soft_labels)
    
    neg_indices = [j for j, label in enumerate(hard_labels) if label == 0]
    if not neg_indices:
        return True
    
    return False


def convert_soft_to_hard(soft_labels):
    """Convert soft labels to 0/1 based on lowest loss"""
    min_idx = soft_labels.index(min(soft_labels))
    hard_labels = [0] * len(soft_labels)
    hard_labels[min_idx] = 1
    return hard_labels


def encode_texts(texts, model, tokenizer, max_length=512):
    """Encode texts with BERT"""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    outputs = model(**inputs)
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
    sum_embeddings = masked_embeddings.sum(dim=1)
    count_tokens = attention_mask.sum(dim=1, keepdim=True)
    embeddings = sum_embeddings / count_tokens
    return F.normalize(embeddings, dim=-1)


def generate_answer(query, context, llm, llm_tokenizer, max_length=900, max_tokens=40):
    """Generate answer autoregressively with stopping criteria"""
    prompt = f"Question: {query} Context: {context} Answer:"
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    prompt_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = llm.generate(
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


def compute_f1(prediction, reference):
    """Compute token-level F1 score"""
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


def compute_exact_match(prediction, reference):
    """Compute exact match score"""
    return float(prediction.lower().strip() == reference.lower().strip())


def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
    try:
        import google.generativeai as genai
        import os

        # Configure the API
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

        # Create the model
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Combine system instruction with prompt if provided
        if system_instruction:
            full_prompt = f"{system_instruction}\n\n{prompt}"
        else:
            full_prompt = prompt

        # Generate response
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"Error: {str(e)}"


def llm_judge_answer(question, reference_answer, generated_answer):
    """Use Gemini to judge if generated answer matches reference answer"""
    prompt = f"""Question: {question}
Reference Answer: {reference_answer}
Generated Answer: {generated_answer}

Does the Generated Answer correctly answer the question in the same way as the Reference Answer? Consider the answers equivalent if they convey the same core information, even if worded differently. The question is provided for additional context.

Answer with only "YES" or "NO":"""

    system_instruction = "You are an expert evaluator. Compare answers for semantic equivalence and respond with only YES or NO."
    
    response = call_llm(prompt, system_instruction)
    response = response.strip().upper()
    
    # Parse response - look for YES/NO
    if "YES" in response:
        return 1.0
    elif "NO" in response:
        return 0.0
    else:
        # Default to 0 if unclear response
        return 0.0


async def llm_judge_answer_async(question, reference_answer, generated_answer):
    """Async version of LLM judge for batch processing"""
    prompt = f"""Question: {question}
Reference Answer: {reference_answer}
Generated Answer: {generated_answer}

Does the Generated Answer correctly answer the question in the same way as the Reference Answer? Consider the answers equivalent if they convey the same core information, even if worded differently. The question is provided for additional context.

Answer with only "YES" or "NO":"""

    system_instruction = "You are an expert evaluator. Compare answers for semantic equivalence and respond with only YES or NO."
    
    try:
        response = await call_llm_async(prompt, system_instruction)
        response = response.strip().upper()
        
        # Parse response - look for YES/NO
        if "YES" in response:
            return 1.0
        elif "NO" in response:
            return 0.0
        else:
            return 0.0
    except Exception as e:
        print(f"Async LLM judge error: {e}")
        return 0.0


async def call_llm_async(prompt, system_instruction=None):
    """Async version of call_llm for parallel API calls"""
    import google.generativeai as genai
    
    # Configure the API
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    
    # Create the model
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Combine system instruction with prompt if provided
    if system_instruction:
        full_prompt = f"{system_instruction}\n\n{prompt}"
    else:
        full_prompt = prompt
    
    # Use asyncio to run the sync function in thread pool
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        response = await loop.run_in_executor(executor, model.generate_content, full_prompt)
        return response.text


async def batch_llm_judge(judge_tasks):
    """Process multiple LLM judge tasks in parallel"""
    results = await asyncio.gather(*judge_tasks, return_exceptions=True)
    return [r if not isinstance(r, Exception) else 0.0 for r in results]


def generate_answers_batch(question_context_pairs, llm, llm_tokenizer, max_length=900, max_tokens=40, batch_size=8):
    """Generate answers for multiple question-context pairs in batches"""
    all_answers = []
    
    # Save original padding side and set to left for decoder-only models
    original_padding_side = llm_tokenizer.padding_side
    llm_tokenizer.padding_side = 'left'
    
    try:
        for i in range(0, len(question_context_pairs), batch_size):
            batch_pairs = question_context_pairs[i:i + batch_size]
            batch_prompts = [f"Question: {q} Context: {c} Answer:" for q, c in batch_pairs]
            
            # Batch tokenize with left padding
            inputs = llm_tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            
            with torch.no_grad():
                outputs = llm.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=llm_tokenizer.eos_token_id,
                    eos_token_id=llm_tokenizer.eos_token_id
                )
            
            # Decode each answer (only the newly generated tokens)
            for j, (input_ids, output) in enumerate(zip(inputs["input_ids"], outputs)):
                input_length = len(input_ids)
                generated_tokens = output[input_length:]
                generated_text = llm_tokenizer.decode(generated_tokens, skip_special_tokens=True)
                all_answers.append(generated_text.strip())
    
    finally:
        # Restore original padding side
        llm_tokenizer.padding_side = original_padding_side
    
    return all_answers


def save_model(model, config):
    """Save trained model"""
    if not config["save_model"]:
        return
    
    save_path = config["model_save_path"] + config["run_name"]
    print(f"Saving model to {save_path}")
    
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"Model saved successfully")


def precompute_squad_embeddings(corpus_texts, model, bert_tokenizer, config):
    """Precompute embeddings for SQuAD corpus"""
    embeddings = []
    model.eval()
    print("Encoding SQuAD")
    
    with torch.no_grad():
        for text in corpus_texts:
            emb = encode_texts([f"passage: {text}"], model, bert_tokenizer, config["encode_max_length"])
            embeddings.append(emb.cpu())
    
    return torch.cat(embeddings, dim=0).cuda()


def evaluate_squad(model, qa_data, corpus_embeddings, squad_corpus, llm, llm_tokenizer, bert_tokenizer, config, num_examples=500):
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
            best_context = squad_corpus[best_context_idx]
            
            retrieval_hit = (best_context_idx == correct_context_idx)
            retrieval_hits.append(retrieval_hit)
            
            # Compute LLM loss
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


def validation_loop_squad(model, squad_qa_data, squad_corpus, llm, llm_tokenizer, bert_tokenizer, config):
    """Run validation on SQuAD subset"""
    squad_embeddings = precompute_squad_embeddings(squad_corpus, model, bert_tokenizer, config)
    squad_results = evaluate_squad(model, squad_qa_data[:config["squad_eval_examples"]], squad_embeddings, squad_corpus, llm, llm_tokenizer, bert_tokenizer, config)
    print("SQuAD validation:", squad_results)
    return squad_results


def validation_loop_msmarco(model, msmarco_qa_data, msmarco_corpus, llm, llm_tokenizer, bert_tokenizer, config):
    """Run validation on MS MARCO subset"""
    msmarco_embeddings = precompute_squad_embeddings(msmarco_corpus, model, bert_tokenizer, config)
    msmarco_results = evaluate_squad(model, msmarco_qa_data, msmarco_embeddings, msmarco_corpus, llm, llm_tokenizer, bert_tokenizer, config)
    print("MS MARCO validation:", msmarco_results)
    return msmarco_results


def run_all_validation(model, squad_qa_data, squad_corpus, msmarco_qa_data, msmarco_corpus, llm, llm_tokenizer, bert_tokenizer, config):
    """Run both SQuAD and MS MARCO validation"""
    # Move LLM to GPU for validation
    llm.to(device)
    
    try:
        squad_results = validation_loop_squad(model, squad_qa_data, squad_corpus, llm, llm_tokenizer, bert_tokenizer, config)
        msmarco_results = validation_loop_msmarco(model, msmarco_qa_data, msmarco_corpus, llm, llm_tokenizer, bert_tokenizer, config)
        
        return {
            "squad": squad_results,
            "msmarco": msmarco_results
        }
    finally:
        # Move LLM back to CPU and clear cache
        llm.cpu()
        torch.cuda.empty_cache()


def train_standard_infonce(soft_train_data, squad_qa_data, squad_corpus, msmarco_qa_data, msmarco_corpus, llm, llm_tokenizer, bert_tokenizer, config):
    """Train Standard InfoNCE on original labels"""
    print("Training Standard InfoNCE on original labels")
    
    model = AutoModel.from_pretrained(config["encoder_model"], trust_remote_code=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    train_step = 0
    
    # Warmup scheduler
    def lr_lambda(step):
        if step < config["warmup_steps"]:
            return step / config["warmup_steps"]
        else:
            return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    validation_results = []
    
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        
        for i in tqdm(range(0, len(soft_train_data), config["batch_size"]), desc=f"Epoch {epoch+1}"):
            batch_items = soft_train_data[i:i+config["batch_size"]]
            batch_loss = 0
            valid_items = 0
            
            # Filter valid items and collect data for batching
            valid_batch_items = [item for item in batch_items if not should_skip_item(item)]
            if len(valid_batch_items) == 0:
                continue
            
            # Collect all queries and passages for batch encoding
            all_queries = []
            all_pos_passages = []
            all_neg_passages = []
            item_metadata = []  # Track which passages belong to which item
            
            for item in valid_batch_items:
                query = item['query']
                passages = item['passages']['passage_text']
                hard_labels = item['passages']['is_selected']
                
                pos_idx = hard_labels.index(1) if 1 in hard_labels else 0
                neg_indices = [j for j, label in enumerate(hard_labels) if label == 0]
                if not neg_indices:
                    continue
                
                valid_items += 1
                
                # Collect for batch encoding
                all_queries.append(f"query: {query}")
                all_pos_passages.append(f"passage: {passages[pos_idx]}")
                
                # Store negative passages and metadata
                neg_passages_for_item = [f"passage: {passages[i]}" for i in neg_indices]
                all_neg_passages.extend(neg_passages_for_item)
                
                item_metadata.append({
                    'neg_start_idx': len(all_neg_passages) - len(neg_passages_for_item),
                    'neg_count': len(neg_passages_for_item)
                })
            
            if valid_items == 0:
                continue
            
            # Batch encoding
            query_embs = encode_texts(all_queries, model, bert_tokenizer, config["encode_max_length"]).to(device)
            pos_embs = encode_texts(all_pos_passages, model, bert_tokenizer, config["encode_max_length"]).to(device)
            neg_embs = encode_texts(all_neg_passages, model, bert_tokenizer, config["encode_max_length"]).to(device)
            
            # Loss computation
            batch_loss = 0
            
            for i, metadata in enumerate(item_metadata):
                # Get embeddings for this item
                query_emb = query_embs[i:i+1]
                pos_emb = pos_embs[i:i+1]
                
                # Get negative embeddings for this item
                neg_start = metadata['neg_start_idx']
                neg_end = neg_start + metadata['neg_count']
                item_neg_embs = neg_embs[neg_start:neg_end]
                
                # Compute similarities
                pos_sim = torch.sum(query_emb * pos_emb, dim=1)
                neg_sims = torch.sum(query_emb.unsqueeze(1) * item_neg_embs, dim=2).squeeze(0)
                
                # InfoNCE loss
                logits = torch.cat([pos_sim, neg_sims])
                logits = logits / config["temperature"]
                labels = torch.zeros(1, dtype=torch.long).to(device)
                loss = F.cross_entropy(logits.unsqueeze(0), labels)
                
                batch_loss += loss
            
            # Normalize by number of valid items
            if valid_items > 0:
                batch_loss = batch_loss / valid_items
            
            wandb.log({"Contrastive loss": batch_loss}, step=train_step)
            
            # Complete backward pass first to free gradients
            if batch_loss > 0:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += batch_loss.item()
            
            # Now do validation with freed GPU memory
            if train_step % config["validation_frequency"] == 0:
                model.eval()
                val_results = run_all_validation(model, squad_qa_data, squad_corpus, msmarco_qa_data, msmarco_corpus, llm, llm_tokenizer, bert_tokenizer, config)
                validation_results.append(val_results)
                model.train()
                
                val_metrics = {
                    "SQuAD F1": val_results['squad']['F1_Score'],
                    "SQuAD EM": val_results['squad']['Exact_Match'],
                    "SQuAD Retrieval": val_results['squad']['Retrieval_Accuracy'],
                    "SQuAD Loss": val_results['squad']['LLM_Loss'],
                    "SQuAD LLM_Judge": val_results['squad']['LLM_Judge'],
                    "MSMARCO F1": val_results['msmarco']['F1_Score'],
                    "MSMARCO EM": val_results['msmarco']['Exact_Match'],
                    "MSMARCO Retrieval": val_results['msmarco']['Retrieval_Accuracy'],
                    "MSMARCO Loss": val_results['msmarco']['LLM_Loss'],
                    "MSMARCO LLM_Judge": val_results['msmarco']['LLM_Judge'],
                }
                wandb.log(val_metrics, step=train_step)
            
            train_step += 1
        
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
        print("VALIDATION", validation_results)
    
    save_model(model, config)
    return model


def train_converted_infonce(soft_train_data, squad_qa_data, squad_corpus, msmarco_qa_data, msmarco_corpus, llm, llm_tokenizer, bert_tokenizer, config):
    """Train Converted InfoNCE on converted labels"""
    print("Training Converted InfoNCE on converted labels")
    
    model = AutoModel.from_pretrained(config["encoder_model"], trust_remote_code=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    train_step = 0
    
    # Warmup scheduler
    def lr_lambda(step):
        if step < config["warmup_steps"]:
            return step / config["warmup_steps"]
        else:
            return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    validation_results = []
    
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        
        for i in tqdm(range(0, len(soft_train_data), config["batch_size"]), desc=f"Epoch {epoch+1}"):
            batch_items = soft_train_data[i:i+config["batch_size"]]
            batch_loss = 0
            
            for item in batch_items:
                if should_skip_item(item):
                    continue
                
                query = item['query']
                passages = item['passages']['passage_text']
                soft_labels = item['passages']['soft_labels']
                
                # Convert soft to hard labels
                hard_labels = convert_soft_to_hard(soft_labels)
                
                pos_idx = hard_labels.index(1)
                neg_indices = [j for j, label in enumerate(hard_labels) if label == 0]
                if not neg_indices:
                    continue
                
                query_emb = encode_texts([f"query: {query}"], model, bert_tokenizer, config["encode_max_length"])
                pos_emb = encode_texts([f"passage: {passages[pos_idx]}"], model, bert_tokenizer, config["encode_max_length"])
                neg_embs = encode_texts([f"passage: {passages[i]}" for i in neg_indices], model, bert_tokenizer, config["encode_max_length"])
                
                pos_sim = torch.sum(query_emb * pos_emb, dim=1)
                neg_sims = torch.sum(query_emb.unsqueeze(1) * neg_embs, dim=2).squeeze(0)
                
                logits = torch.cat([pos_sim, neg_sims])
                logits = logits / config["temperature"]
                labels = torch.zeros(1, dtype=torch.long).to(device)
                loss = F.cross_entropy(logits.unsqueeze(0), labels)
                
                batch_loss += loss
            
            wandb.log({"Contrastive loss": batch_loss}, step=train_step)
            
            if train_step % config["validation_frequency"] == 0:
                model.eval()
                val_results = run_all_validation(model, squad_qa_data, squad_corpus, msmarco_qa_data, msmarco_corpus, llm, llm_tokenizer, bert_tokenizer, config)
                validation_results.append(val_results)
                model.train()
                
                val_metrics = {
                    "SQuAD F1": val_results['squad']['F1_Score'],
                    "SQuAD EM": val_results['squad']['Exact_Match'],
                    "SQuAD Retrieval": val_results['squad']['Retrieval_Accuracy'],
                    "SQuAD Loss": val_results['squad']['LLM_Loss'],
                    "SQuAD LLM_Judge": val_results['squad']['LLM_Judge'],
                    "MSMARCO F1": val_results['msmarco']['F1_Score'],
                    "MSMARCO EM": val_results['msmarco']['Exact_Match'],
                    "MSMARCO Retrieval": val_results['msmarco']['Retrieval_Accuracy'],
                    "MSMARCO Loss": val_results['msmarco']['LLM_Loss'],
                    "MSMARCO LLM_Judge": val_results['msmarco']['LLM_Judge'],
                }
                wandb.log(val_metrics, step=train_step)
            
            if batch_loss > 0:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += batch_loss.item()
            
            train_step += 1
        
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
        print("VALIDATION", validation_results)
    
    save_model(model, config)
    return model


def train_kl_soft_infonce_batched(soft_train_data, squad_qa_data, squad_corpus, msmarco_qa_data, msmarco_corpus, llm, llm_tokenizer, bert_tokenizer, config):
    """Train batched KL Soft InfoNCE"""
    print(f"Training batched KL Soft InfoNCE (teacher_temp={config['teacher_temp']}, student_temp={config['student_temp']})")
    
    model = AutoModel.from_pretrained(config["encoder_model"], trust_remote_code=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    train_step = 0
    validation_results = []
    
    # Warmup scheduler
    def lr_lambda(step):
        if step < config["warmup_steps"]:
            return step / config["warmup_steps"]
        else:
            return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        
        for i in tqdm(range(0, len(soft_train_data), config["batch_size"]), desc=f"Epoch {epoch+1}"):
            batch_items = soft_train_data[i:i+config["batch_size"]]
            batch_items = [item for item in batch_items if not should_skip_item(item)]
            if len(batch_items) == 0:
                continue
            
            queries = []
            passages = []
            passage_counts = []
            soft_label_groups = []
            
            for item in batch_items:
                q = item["query"]
                p_list = item["passages"]["passage_text"]
                l_list = item["passages"]["soft_labels"]
                
                queries.append(f"query: {q}")
                passages.extend([f"passage: {p}" for p in p_list])
                soft_label_groups.append(torch.tensor(l_list, dtype=torch.float32, device=device))
                passage_counts.append(len(p_list))
            
            query_embs = encode_texts(queries, model, bert_tokenizer, config["encode_max_length"]).to(device)
            passage_embs = encode_texts(passages, model, bert_tokenizer, config["encode_max_length"]).to(device)
            passage_emb_groups = torch.split(passage_embs, passage_counts, dim=0)
            
            batch_loss = 0
            for q_emb, p_embs, soft_labels in zip(query_embs, passage_emb_groups, soft_label_groups):
                similarities = F.cosine_similarity(q_emb.unsqueeze(0), p_embs, dim=1)
                
                teacher_probs = F.softmax(-soft_labels / config["teacher_temp"], dim=0)
                log_student_probs = F.log_softmax(similarities / config["student_temp"], dim=0)
                
                loss = F.kl_div(log_student_probs, teacher_probs, reduction='batchmean')
                
                # Add margin hinge loss
                positive_idx = soft_labels.argmin()
                positive_sim = similarities[positive_idx]
                negative_sims = torch.cat([similarities[:positive_idx], similarities[positive_idx+1:]])
                alpha = 1.0
                hinge_loss = torch.clamp(config["margin"] - (positive_sim - negative_sims), min=0).mean()
                
                loss = loss + alpha * hinge_loss
                batch_loss += loss
            
            batch_loss = batch_loss / len(batch_items)
            
            wandb.log({"KL loss": batch_loss}, step=train_step)
            
            # Complete backward pass first to free gradients
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += batch_loss.item()
            
            # Now do validation with freed GPU memory
            if train_step % config["validation_frequency"] == 0:
                model.eval()
                val_results = run_all_validation(model, squad_qa_data, squad_corpus, msmarco_qa_data, msmarco_corpus, llm, llm_tokenizer, bert_tokenizer, config)
                validation_results.append(val_results)
                model.train()
                
                val_metrics = {
                    "SQuAD F1": val_results['squad']['F1_Score'],
                    "SQuAD EM": val_results['squad']['Exact_Match'],
                    "SQuAD Retrieval": val_results['squad']['Retrieval_Accuracy'],
                    "SQuAD Loss": val_results['squad']['LLM_Loss'],
                    "SQuAD LLM_Judge": val_results['squad']['LLM_Judge'],
                    "MSMARCO F1": val_results['msmarco']['F1_Score'],
                    "MSMARCO EM": val_results['msmarco']['Exact_Match'],
                    "MSMARCO Retrieval": val_results['msmarco']['Retrieval_Accuracy'],
                    "MSMARCO Loss": val_results['msmarco']['LLM_Loss'],
                    "MSMARCO LLM_Judge": val_results['msmarco']['LLM_Judge'],
                }
                wandb.log(val_metrics, step=train_step)
            
            train_step += 1
        
        print(f"Epoch {epoch+1} total loss: {total_loss:.4f}")
        print("VALIDATION", validation_results)
    
    save_model(model, config)
    return model


def main():
    """Main training script"""
    # Get config
    config = get_config()
    
    # Setup wandb
    if config["wandb_key"]:
        print(f"Logging into wandb with key: {config['wandb_key'][:8]}...")
        wandb.login(key=config["wandb_key"])
    else:
        print("No wandb API key found - skipping automatic login")
    
    # Setup
    global device
    device, bert_tokenizer, llm, llm_tokenizer = setup_device_and_models(config)
    
    # Load datasets
    print("Loading SQuAD dataset...")
    squad_dataset = load_dataset("squad_v2")
    squad_train = squad_dataset["train"]
    
    random.seed(42)
    squad_qa_data, squad_corpus, title_contexts = preprocess_squad_title_diverse(
        squad_train, num_titles=config["squad_num_titles"], questions_per_title=config["squad_questions_per_title"]
    )
    
    # Load soft-labeled data from HuggingFace
    print("Loading MS MARCO soft-labeled dataset...")
    ms_marco_dataset = load_dataset(config["dataset_name"])
    soft_train_data_full = list(ms_marco_dataset["train"])
    
    if config["num_data_examples"] is None or config["num_data_examples"] == -1:
        soft_train_data = soft_train_data_full
        print(f"Using all {len(soft_train_data)} training examples")
    else:
        soft_train_data = soft_train_data_full[:config["num_data_examples"]]
        print(f"Using {len(soft_train_data)} training examples")
    
    # Prepare MS MARCO validation data
    msmarco_qa_data, msmarco_corpus = preprocess_msmarco_validation(config, config["msmarco_val_examples"])
    print(f"MS MARCO validation corpus: {len(msmarco_corpus)} contexts")
    
    wandb.init(
        project=config["wandb_project"],
        name=config["run_name"],
        config=config
    )
    
    # Train model based on method
    if config["training_method"] == "standard_infonce":
        model = train_standard_infonce(soft_train_data, squad_qa_data, squad_corpus, msmarco_qa_data, msmarco_corpus, llm, llm_tokenizer, bert_tokenizer, config)
    elif config["training_method"] == "converted_infonce":
        model = train_converted_infonce(soft_train_data, squad_qa_data, squad_corpus, msmarco_qa_data, msmarco_corpus, llm, llm_tokenizer, bert_tokenizer, config)
    elif config["training_method"] == "kl_soft_infonce":
        model = train_kl_soft_infonce_batched(soft_train_data, squad_qa_data, squad_corpus, msmarco_qa_data, msmarco_corpus, llm, llm_tokenizer, bert_tokenizer, config)
    else:
        raise ValueError(f"Unknown training method: {config['training_method']}")
    
    # Final evaluation
    squad_test_start = config["squad_eval_examples"]
    squad_test_end = squad_test_start + config["squad_test_examples"]
    squad_test_set = squad_qa_data[squad_test_start:squad_test_end]
    
    squad_embeddings = precompute_squad_embeddings(squad_corpus, model, bert_tokenizer, config)
    squad_results = evaluate_squad(model, squad_test_set, squad_embeddings, squad_corpus, llm, llm_tokenizer, bert_tokenizer, config)
    
    # MS MARCO final evaluation
    msmarco_test_qa_data, msmarco_test_corpus = preprocess_msmarco_validation(config, config["msmarco_test_examples"])
    msmarco_embeddings = precompute_squad_embeddings(msmarco_test_corpus, model, bert_tokenizer, config)
    msmarco_results = evaluate_squad(model, msmarco_test_qa_data, msmarco_embeddings, msmarco_test_corpus, llm, llm_tokenizer, bert_tokenizer, config)
    
    # Log final results
    final_results = {
        "Final_SQuAD_F1": squad_results['F1_Score'],
        "Final_SQuAD_EM": squad_results['Exact_Match'],
        "Final_SQuAD_Retrieval": squad_results['Retrieval_Accuracy'],
        "Final_SQuAD_LLM_Loss": squad_results['LLM_Loss'],
        "Final_SQuAD_LLM_Judge": squad_results['LLM_Judge'],
        "Final_MSMARCO_F1": msmarco_results['F1_Score'],
        "Final_MSMARCO_EM": msmarco_results['Exact_Match'],
        "Final_MSMARCO_Retrieval": msmarco_results['Retrieval_Accuracy'],
        "Final_MSMARCO_LLM_Loss": msmarco_results['LLM_Loss'],
        "Final_MSMARCO_LLM_Judge": msmarco_results['LLM_Judge'],
    }
    wandb.log(final_results)
    
    print("Final SQuAD results:", squad_results)
    print("Final MS MARCO results:", msmarco_results)
    
    # Cleanup
    del model, squad_embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main() 
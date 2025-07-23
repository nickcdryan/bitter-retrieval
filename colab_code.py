!pip install -U datasets huggingface_hub fsspec
!pip install -U datasets huggingface_hub fsspec
!pip install wandb
from datasets import load_dataset
import random
from collections import defaultdict


!pip install -U datasets huggingface_hub fsspec
!pip install wandb
import wandb
from google.colab import userdata
wandb.login(key=userdata.get('WANDB_KEY'))
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm
import json







# Load and preprocess SQuAD data
print("Loading SQuAD dataset...")
squad_dataset = load_dataset("squad_v2")
squad_validation = squad_dataset["validation"]
squad_train = squad_dataset["train"]


def preprocess_squad_title_diverse(data, num_titles=50, questions_per_title=5):
    """
    Version 2: Take a few questions from each title but include ALL contexts for selected titles
    This creates harder retrieval since there are multiple similar contexts per topic
    """
    # Group data by title
    title_groups = defaultdict(list)
    for item in data:
        title = item.get("title", "Unknown")
        title_groups[title].append(item)

    print(f"Found {len(title_groups)} unique titles in dataset")

    # Randomly select titles to include
    selected_titles = random.sample(list(title_groups.keys()), min(num_titles, len(title_groups)))

    qa_data = []
    corpus_texts = []
    context_to_idx = {}
    title_to_contexts = defaultdict(set)  # Track which contexts belong to which title

    # For each selected title
    for title in selected_titles:
        title_items = title_groups[title]

        # First pass: collect ALL unique contexts for this title
        title_contexts = set()
        for item in title_items:
            title_contexts.add(item["context"])

        # Add all contexts from this title to corpus
        for context in title_contexts:
            if context not in context_to_idx:
                context_to_idx[context] = len(corpus_texts)
                corpus_texts.append(context)
                title_to_contexts[title].add(len(corpus_texts) - 1)

        # Second pass: sample questions from this title
        sampled_questions = random.sample(title_items, min(questions_per_title, len(title_items)))

        for item in sampled_questions:
            context = item["context"]
            question = item["question"]
            answers = item["answers"]

            # Format answers
            if answers["text"]:  # Answerable question
                answer = answers["text"][0]  # Take first answer
            else:  # Unanswerable question
                answer = "Unanswerable"

            qa_data.append({
                "question": question,
                "answer": answer,
                "context_idx": context_to_idx[context],
                "title": title,
                "correct_context": context,
                "title_context_indices": list(title_to_contexts[title])  # All context indices for this title
            })

    # Print statistics
    contexts_per_title = [len(contexts) for contexts in title_to_contexts.values()]
    print(f"Selected {len(selected_titles)} titles")
    print(f"Total questions: {len(qa_data)}")
    print(f"Total unique contexts: {len(corpus_texts)}")
    print(f"Avg contexts per title: {sum(contexts_per_title) / len(contexts_per_title):.1f}")
    print(f"Min/Max contexts per title: {min(contexts_per_title)}/{max(contexts_per_title)}")

    return qa_data, corpus_texts, title_to_contexts

random.seed(42)  # For reproducibility
squad_qa_data, squad_corpus, title_contexts = preprocess_squad_title_diverse(
    squad_train,
    num_titles=100,
    questions_per_title=5
)

# Load soft-labeled data
fp = 'drive/MyDrive/Colab_Notebooks/experiments/retrieval_network/'
filename = 'soft_labels_msmarco_llama3B_0-25000.json'
full_fp = fp + filename
with open(full_fp, "r") as f:
    soft_train_data_full = json.load(f)


# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2 # 16
learning_rate = 2e-5
num_epochs = 2

soft_data = soft_train_data_full

soft_train_data = soft_train_data_full[:25000]
soft_eval_data = soft_train_data_full[4000:4200]

# Load models
# bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
bert_tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1-unsupervised", trust_remote_code=True)

llm_path = "meta-llama/Llama-3.2-3B"
llm = AutoModelForCausalLM.from_pretrained(llm_path).to(device).eval()
llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)

if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token


def should_skip_item(item):
    if not item['answers']:
        return True
    if len(item['passages']['passage_text']) < 2:
        return True

    # For standard/converted: check if negatives exist
    if hasattr(item['passages'], 'is_selected'):
        hard_labels = item['passages']['is_selected']
    else:
        # For soft method: convert and check
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


# Evaluation 1: Ranking metrics
def evaluate_ranking(model, label_type="original"):
    """Evaluate ranking performance"""
    model.eval()
    ranks = []

    with torch.no_grad():
        for item in tqdm(soft_eval_data, desc=f"Ranking eval ({label_type})"):
            if not item['answers']:
                continue

            query = item['query']
            passages = item['passages']['passage_text']

            if label_type == "original":
                labels = item['passages']['is_selected']
                pos_idx = labels.index(1) if 1 in labels else None
            else:  # converted
                soft_labels = item['passages']['soft_labels']
                labels = convert_soft_to_hard(soft_labels)
                pos_idx = labels.index(1)

            if pos_idx is None:
                continue

            # # Encode and compute similarities
            query_emb = encode_texts([f"query: {query}"], model, bert_tokenizer)
            passage_embs = encode_texts([f"passage: {p}" for p in passages], model, bert_tokenizer)
            similarities = torch.matmul(query_emb, passage_embs.T).squeeze(0)

            # Find rank of positive passage
            sorted_indices = torch.argsort(similarities, descending=True)
            rank = (sorted_indices == pos_idx).nonzero(as_tuple=True)[0].item()
            ranks.append(rank)

    ranks = np.array(ranks)
    return {
        "MRR": np.mean(1 / (ranks + 1)),
        "Recall@1": (ranks == 0).mean(),
        "Recall@5": (ranks < 5).mean(),
    }

# Evaluation 2: Average LLM loss
def evaluate_llm_loss(model):
    """Evaluate average loss on answer tokens"""
    model.eval()
    all_losses = []

    with torch.no_grad():
        for item in tqdm(soft_eval_data, desc="LLM loss eval"):
            if not item['answers']:
                continue

            query = item['query']
            answer = item['answers'][0]
            passages = item['passages']['passage_text']

            # Encode and find best passage according to BERT
            query_emb = encode_texts([f"query: {query}"], model, bert_tokenizer)
            passage_embs = encode_texts([f"passage: {p}" for p in passages], model, bert_tokenizer)

            similarities = torch.matmul(query_emb, passage_embs.T).squeeze(0)
            best_passage_idx = similarities.argmax().item()
            best_passage = passages[best_passage_idx]

            # Compute LLM loss for best passage
            input_text = f"Question: {query} Context: {best_passage} Answer: {answer}"
            inputs = llm_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
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
                all_losses.append(loss.item())

    return {"Avg_LLM_Loss": np.mean(all_losses)}




# Add generation evaluation functions
def generate_answer(query, context, max_tokens=40):
    """Generate answer autoregressively with stopping criteria"""
    prompt = f"Question: {query} Context: {context} Answer:"
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900).to(device)
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
    #print ("RAW ANSWER", generated_text)

    # if ". " in generated_text:
    #     generated_text = generated_text.split(". ")[0]

    return generated_text.strip()

def compute_f1(prediction, reference):
    """Compute token-level F1 score"""
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

def compute_exact_match(prediction, reference):
    """Compute exact match score"""
    return float(prediction.lower().strip() == reference.lower().strip())

def evaluate_generation(model, test_data, num_examples=50):
    """Evaluate answer generation quality"""
    model.eval()
    exact_matches = []
    f1_scores = []

    with torch.no_grad():
        for item in tqdm(test_data[:num_examples], desc="Generation eval"):
            if not item['answers']:
                continue

            query = item['query']
            reference_answer = item['answers'][0]
            passages = item['passages']['passage_text']

            # query_emb = encode_texts([query], model, bert_tokenizer)
            # passage_embs = encode_texts(passages, model, bert_tokenizer)

            query_emb = encode_texts([f"query: {query}"], model, bert_tokenizer)
            passage_embs = encode_texts([f"passage: {p}" for p in passages], model, bert_tokenizer)
            similarities = torch.matmul(query_emb, passage_embs.T).squeeze(0)
            best_passage = passages[similarities.argmax().item()]

            try:
                generated_answer = generate_answer(query, best_passage)
                #print ("\n\n GEN: ", generated_answer, "\nREF: ", reference_answer, "\n\n")
                em = compute_exact_match(generated_answer, reference_answer)
                f1 = compute_f1(generated_answer, reference_answer)
                exact_matches.append(em)
                f1_scores.append(f1)
            except:
                continue

    return {"Exact_Match": np.mean(exact_matches), "F1_Score": np.mean(f1_scores)}

def encode_texts(texts, model, tokenizer):
    """Encode texts with BERT"""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model(**inputs)
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
    sum_embeddings = masked_embeddings.sum(dim=1)
    count_tokens = attention_mask.sum(dim=1, keepdim=True)
    embeddings = sum_embeddings / count_tokens
    return F.normalize(embeddings, dim=-1)



def precompute_squad_embeddings(corpus_texts, model):
    embeddings = []
    model.eval()  # Set to eval mode
    print ("Encoding SQuAD")

    with torch.no_grad():  # This is crucial!
        for text in corpus_texts:
            emb = encode_texts([f"passage: {text}"], model, bert_tokenizer)
            embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0).cuda()

def evaluate_squad(model, qa_data, corpus_embeddings, num_examples=500):
    """Evaluate retrieval and generation on SQuAD"""
    model.eval()

    retrieval_hits = []  # Did we retrieve the correct context?
    llm_losses = []
    exact_matches = []
    f1_scores = []

    with torch.no_grad():
        for item in tqdm(qa_data[:num_examples], desc="SQuAD evaluation"):
            if item["answer"] == "Unanswerable":
                continue

            question = item["question"]
            correct_answer = item["answer"]
            correct_context_idx = item["context_idx"]

            # Encode question and retrieve top context
            question_emb = encode_texts([f"query: {question}"], model, bert_tokenizer)
            similarities = torch.matmul(question_emb, corpus_embeddings.T).squeeze(0)
            best_context_idx = similarities.argmax().item()
            best_context = squad_corpus[best_context_idx]

            # Check if we retrieved the correct context
            retrieval_hit = (best_context_idx == correct_context_idx)
            retrieval_hits.append(retrieval_hit)

            # Compute LLM loss with retrieved context
            input_text = f"Question: {question} Context: {best_context} Answer: {correct_answer}"
            inputs = llm_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
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

            # Generate answer and compute EM/F1
            try:
                generated_answer = generate_answer(question, best_context)
                em = compute_exact_match(generated_answer, correct_answer)
                f1 = compute_f1(generated_answer, correct_answer)
                exact_matches.append(em)
                f1_scores.append(f1)
            except:
                continue

    return {
        "Retrieval_Accuracy": np.mean(retrieval_hits),
        "LLM_Loss": np.mean(llm_losses),
        "Exact_Match": np.mean(exact_matches),
        "F1_Score": np.mean(f1_scores),
        "Num_Examples": len(retrieval_hits)
    }


def validation_loop_squad(model):
    global squad_qa_data
    global squad_corpus
    squad_embeddings = precompute_squad_embeddings(squad_corpus, model)
    squad_results = evaluate_squad(model, squad_qa_data[:100], squad_embeddings)
    print (squad_results)
    return squad_results

# Method 1: Standard InfoNCE on original labels
def train_standard_infonce(temp=1.0):
    print("Training Standard InfoNCE on original labels")
    # model = AutoModel.from_pretrained("google-bert/bert-base-uncased").to(device)
    model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1-unsupervised", trust_remote_code=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_step = 0

    # WARMUP
    warmup_steps = 400
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup from 0 to 1 over warmup_steps
            return step / warmup_steps
        else:
            # After warmup, keep learning rate constant
            return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # WARMUP

    validation_results = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for i in tqdm(range(0, len(soft_train_data), batch_size), desc=f"Epoch {epoch+1}"):
            batch_items = soft_train_data[i:i+batch_size]
            batch_loss = 0

            for item in batch_items:
                if should_skip_item(item):
                    continue

                query = item['query']
                passages = item['passages']['passage_text']
                hard_labels = item['passages']['is_selected']

                # Find positive and negative
                pos_idx = hard_labels.index(1) if 1 in hard_labels else 0
                neg_indices = [j for j, label in enumerate(hard_labels) if label == 0]
                if not neg_indices:
                    continue

                query_emb = encode_texts([f"query: {query}"], model, bert_tokenizer)
                pos_emb = encode_texts([f"passage: {passages[pos_idx]}"], model, bert_tokenizer)
                neg_embs = encode_texts([f"passage: {passages[i]}" for i in neg_indices], model, bert_tokenizer)

                # Compute similarities
                pos_sim = torch.sum(query_emb * pos_emb, dim=1)
                neg_sims = torch.sum(query_emb.unsqueeze(1) * neg_embs, dim=2).squeeze(0)

                # InfoNCE loss with all negatives
                logits = torch.cat([pos_sim, neg_sims])
                # add temperature
                logits = logits / temp
                labels = torch.zeros(1, dtype=torch.long).to(device)  # First is positive
                loss = F.cross_entropy(logits.unsqueeze(0), labels)


                batch_loss += loss

            metrics = {
            "Contrastive loss": batch_loss,
            }
            # Log all metrics at once
            wandb.log(metrics, step=train_step)#i

            if train_step % 750 == 0:
                model.eval()
                val_results = validation_loop_squad(model)
                print (val_results)
                validation_results.append(val_results)
                model.train()
                val_metrics = {
                    "SQuAD F1": val_results['F1_Score'],
                    "SQuAD EM": val_results['Exact_Match'],
                    "SQuAD Retrieval": val_results['Retrieval_Accuracy'],
                    "SQuAD Loss": val_results['LLM_Loss'],
                }
                wandb.log(val_metrics, step=train_step)

            if batch_loss > 0:
                #print(f"Batch loss: {batch_loss.item():.4f}")
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                scheduler.step() # WARMUP
                total_loss += batch_loss.item()


            train_step += 1

        print(f"  Epoch {epoch+1} Loss: {total_loss:.4f}")
        print ("VALIDATION", validation_results)

    return model


# Method 2: Converted InfoNCE on converted labels
def train_converted_infonce(temp=1.0):
    print("Training Converted InfoNCE on original labels")
    #model = AutoModel.from_pretrained("google-bert/bert-base-uncased").to(device)
    model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1-unsupervised", trust_remote_code=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_step = 0
    validation_results = []

    # WARMUP
    warmup_steps = 400
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup from 0 to 1 over warmup_steps
            return step / warmup_steps
        else:
            # After warmup, keep learning rate constant
            return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # WARMUP

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for i in tqdm(range(0, len(soft_train_data), batch_size), desc=f"Epoch {epoch+1}"):
            batch_items = soft_train_data[i:i+batch_size]
            batch_loss = 0

            for item in batch_items:
                if should_skip_item(item):
                    continue

                query = item['query']
                passages = item['passages']['passage_text']
                soft_labels = item['passages']['soft_labels']

                # Convert soft to hard labels
                hard_labels = convert_soft_to_hard(soft_labels)

                # Find positive and negative
                pos_idx = hard_labels.index(1)
                neg_indices = [j for j, label in enumerate(hard_labels) if label == 0]
                if not neg_indices:
                    continue

                query_emb = encode_texts([f"query: {query}"], model, bert_tokenizer)
                pos_emb = encode_texts([f"passage: {passages[pos_idx]}"], model, bert_tokenizer)
                neg_embs = encode_texts([f"passage: {passages[i]}" for i in neg_indices], model, bert_tokenizer)

                # Compute similarities
                pos_sim = torch.sum(query_emb * pos_emb, dim=1)
                neg_sims = torch.sum(query_emb.unsqueeze(1) * neg_embs, dim=2).squeeze(0)

                # InfoNCE loss with all negatives
                logits = torch.cat([pos_sim, neg_sims])
                logits = logits / temp
                labels = torch.zeros(1, dtype=torch.long).to(device)  # First is positive
                loss = F.cross_entropy(logits.unsqueeze(0), labels)


                batch_loss += loss

            metrics = {
            "Contrastive loss": batch_loss,
            }
            # Log all metrics at once
            wandb.log(metrics, step=train_step)#i


            if train_step % 750 == 0:
                model.eval()
                val_results = validation_loop_squad(model)
                print (val_results)
                validation_results.append(val_results)
                model.train()
                val_metrics = {
                    "SQuAD F1": val_results['F1_Score'],
                    "SQuAD EM": val_results['Exact_Match'],
                    "SQuAD Retrieval": val_results['Retrieval_Accuracy'],
                    "SQuAD Loss": val_results['LLM_Loss'],
                }
                wandb.log(val_metrics, step=train_step)

            if batch_loss > 0:
                #print(f"Batch loss: {batch_loss.item():.4f}")
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                scheduler.step() # WARMUP
                total_loss += batch_loss.item()

            train_step += 1

        print(f"  Epoch {epoch+1} Loss: {total_loss:.4f}")
        print ("VALIDATION", validation_results)


    return model



def train_kl_soft_infonce_batched(temp_teacher=0.1, temp_student=0.05, margin=1.0):
    print(f"Training batched KL Soft InfoNCE (teacher_temp={temp_teacher}, student_temp={temp_student})")
    model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1-unsupervised", trust_remote_code=True).to(device)
    # model = AutoModel.from_pretrained("google-bert/bert-base-uncased").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_step = 0

    validation_results = []

    # WARMUP
    warmup_steps = 400
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup from 0 to 1 over warmup_steps
            return step / warmup_steps
        else:
            # After warmup, keep learning rate constant
            return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # WARMUP

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for i in tqdm(range(0, len(soft_train_data), batch_size), desc=f"Epoch {epoch+1}"):
            batch_items = soft_train_data[i:i+batch_size]

            # Filter out skipped items ahead of time
            batch_items = [item for item in batch_items if not should_skip_item(item)]
            if len(batch_items) == 0:
                continue

            # ----- PREP DATA -----
            queries = []
            passages = []
            passage_counts = []  # number of passages per query
            soft_label_groups = []

            for item in batch_items:
                q = item["query"]
                p_list = item["passages"]["passage_text"]
                l_list = item["passages"]["soft_labels"]

                queries.append(f"query: {q}")
                passages.extend([f"passage: {p}" for p in p_list])
                soft_label_groups.append(torch.tensor(l_list, dtype=torch.float32, device=device))
                passage_counts.append(len(p_list))

            # ----- ENCODE -----
            query_embs = encode_texts(queries, model, bert_tokenizer).to(device)       # shape (B, D)
            passage_embs = encode_texts(passages, model, bert_tokenizer).to(device)    # shape (sum(K_i), D)

            # ----- SPLIT PASSAGE EMBEDDINGS PER QUERY -----
            passage_emb_groups = torch.split(passage_embs, passage_counts, dim=0)  # list of tensors (K_i, D)

            # ----- LOSS -----
            batch_loss = 0
            for q_emb, p_embs, soft_labels in zip(query_embs, passage_emb_groups, soft_label_groups):
                # Cosine similarities between query and its passages (K,)
                similarities = F.cosine_similarity(q_emb.unsqueeze(0), p_embs, dim=1)

                # Teacher: sharp softmax over usefulness
                teacher_probs = F.softmax(-soft_labels / temp_teacher, dim=0)

                # Student: predicted similarities
                log_student_probs = F.log_softmax(similarities / temp_student, dim=0)

                # KL divergence
                loss = F.kl_div(log_student_probs, teacher_probs, reduction='batchmean')


                # Add margin hinge loss
                positive_idx = soft_labels.argmin()
                positive_sim = similarities[positive_idx]
                negative_sims = torch.cat([similarities[:positive_idx], similarities[positive_idx+1:]])
                alpha = 1.0
                hinge_loss = torch.clamp(margin - (positive_sim - negative_sims), min=0).mean()

                loss = loss + alpha * hinge_loss

                batch_loss += loss

            batch_loss = batch_loss / len(batch_items)  # normalize by batch size

            metrics = {
            "KL loss": batch_loss,
            }
            # Log all metrics at once
            wandb.log(metrics, step=train_step)#i


            if train_step % 750 == 0:
                model.eval()
                val_results = validation_loop_squad(model)
                print (val_results)
                validation_results.append(val_results)
                model.train()
                val_metrics = {
                    "SQuAD F1": val_results['F1_Score'],
                    "SQuAD EM": val_results['Exact_Match'],
                    "SQuAD Retrieval": val_results['Retrieval_Accuracy'],
                    "SQuAD Loss": val_results['LLM_Loss'],
                }
                wandb.log(val_metrics, step=train_step)

            # ----- BACKWARD -----
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += batch_loss.item()

            train_step += 1

        print(f"Epoch {epoch+1} total loss: {total_loss:.4f}")
        print ("VALIDATION", validation_results)

    return model

# Method 1: BERT Baseline (no training)
def eval_bert_baseline():
    #model = AutoModel.from_pretrained("google-bert/bert-base-uncased").to(device)
    model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1-unsupervised", trust_remote_code=True).to(device)
    return model


teacher_temps = [.01]
student_temps = [.01]
squad_test_set = squad_qa_data[100:500]

results = {}
for s in student_temps:
    for t in teacher_temps:
        #run_name = f"kl-margin:1.0-test-5k-2epochs-bs:8-2e-5-teacher:{t}-student:{s}"
        run_name = "standard_infonce-inbatchneg-NOMIC-25k-emptystring:include-temp:.02-epochs:2-bs:2-gradacc:8-LR:2e-5-warmup:400"
        #run_name = f"kl-margin:3.0-NOMIC-25k-emptystring:include-2epochs-bs:8-2e-5-teacher:{t}-student:{s}"
        wandb.init(
            project="bitter-retrieval",
            name=run_name,
            notes="""
            """,

            config={
            }
        )

        #model = train_soft_infonce(sigma=0.2, tau=0.2, temp_teacher=t, temp_student=s)
        #model = train_kl_soft_infonce_batched(temp_teacher=t, temp_student=s, margin=3.0)
        model = train_standard_infonce(temp=.02)
        #model = train_converted_infonce(temp=.02)

        squad_embeddings = precompute_squad_embeddings(squad_corpus, model)
        squad_results = evaluate_squad(model, squad_test_set, squad_embeddings)

        print (squad_results)
        results[f"teacher_{t}_student_{s}"] = squad_results

        del model, squad_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print (results)
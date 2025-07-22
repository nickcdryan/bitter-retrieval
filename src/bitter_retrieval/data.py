"""
Data loading for bitter-retrieval training.
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
from datasets import load_dataset

logger = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_soft_labeled_data(filepath: str) -> List[Dict[str, Any]]:
    """Load soft-labeled MS MARCO data from JSON file."""
    logger.info(f"Loading soft-labeled data from {filepath}")
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Soft-labeled data file not found: {filepath}")
    
    with open(filepath, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} examples from {filepath}")
    return data


def should_skip_item(item: Dict[str, Any]) -> bool:
    """Check if a data item should be skipped during training."""
    # Skip items without answers
    if not item.get('answers'):
        return True
    
    # Skip items with too few passages
    if len(item['passages']['passage_text']) < 2:
        return True

    # Check if we have negative examples
    if 'is_selected' in item['passages']:
        hard_labels = item['passages']['is_selected']
    else:
        soft_labels = item['passages']['soft_labels']
        hard_labels = convert_soft_to_hard(soft_labels)

    # Skip if no negative examples
    neg_indices = [j for j, label in enumerate(hard_labels) if label == 0]
    return len(neg_indices) == 0


def convert_soft_to_hard(soft_labels: List[float]) -> List[int]:
    """Convert soft labels to binary hard labels based on lowest loss."""
    min_idx = soft_labels.index(min(soft_labels))
    hard_labels = [0] * len(soft_labels)
    hard_labels[min_idx] = 1
    return hard_labels


def preprocess_squad_data(
    data, 
    num_titles: int = 50, 
    questions_per_title: int = 5
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, set]]:
    """
    Preprocess SQuAD data for retrieval evaluation.
    Creates diverse contexts per title for challenging retrieval.
    """
    logger.info("Preprocessing SQuAD data for retrieval evaluation")
    
    # Group data by title
    title_groups = defaultdict(list)
    for item in data:
        title = item.get("title", "Unknown")
        title_groups[title].append(item)

    logger.info(f"Found {len(title_groups)} unique titles in dataset")

    # Randomly select titles
    selected_titles = random.sample(
        list(title_groups.keys()), 
        min(num_titles, len(title_groups))
    )

    qa_data = []
    corpus_texts = []
    context_to_idx = {}
    title_to_contexts = defaultdict(set)

    # Process each selected title
    for title in selected_titles:
        title_items = title_groups[title]

        # Collect all unique contexts for this title
        title_contexts = set()
        for item in title_items:
            title_contexts.add(item["context"])

        # Add contexts to corpus
        for context in title_contexts:
            if context not in context_to_idx:
                context_to_idx[context] = len(corpus_texts)
                corpus_texts.append(context)
                title_to_contexts[title].add(len(corpus_texts) - 1)

        # Sample questions from this title
        sampled_questions = random.sample(
            title_items, 
            min(questions_per_title, len(title_items))
        )

        for item in sampled_questions:
            context = item["context"]
            question = item["question"]
            answers = item["answers"]

            # Format answers
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

    logger.info(f"Selected {len(selected_titles)} titles")
    logger.info(f"Total questions: {len(qa_data)}")
    logger.info(f"Total unique contexts: {len(corpus_texts)}")

    return qa_data, corpus_texts, title_to_contexts


def load_train_data(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load training data from MS MARCO soft labels."""
    soft_data = load_soft_labeled_data(config["soft_labels_path"])
    return soft_data[:config["train_size"]]


def load_eval_data(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load evaluation data from MS MARCO soft labels."""
    soft_data = load_soft_labeled_data(config["soft_labels_path"])
    start_idx = config["eval_start_idx"]
    end_idx = start_idx + config["eval_size"]
    return soft_data[start_idx:end_idx]


def load_squad_data(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, set]]:
    """Load and preprocess SQuAD dataset for evaluation."""
    logger.info("Loading SQuAD dataset...")
    
    squad_dataset = load_dataset("squad_v2")
    squad_train = squad_dataset["train"]
    
    # Set random seed for reproducible preprocessing
    set_random_seed(config["seed"])
    
    qa_data, corpus_texts, title_contexts = preprocess_squad_data(
        squad_train,
        num_titles=config["squad_num_titles"],
        questions_per_title=config["squad_questions_per_title"]
    )
    
    return qa_data, corpus_texts, title_contexts


def get_squad_test_split(squad_qa_data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get SQuAD test split."""
    start_idx = config["squad_num_examples"]
    end_idx = start_idx + config["squad_test_size"]
    return squad_qa_data[start_idx:end_idx] 
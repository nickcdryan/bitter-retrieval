"""
MS MARCO Dataset Soft Labeling with LLM

This module generates soft labels for MS MARCO dataset using a Large Language Model.
The soft labels represent LLM-computed losses for passage-answer pairs.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for the soft labeling process."""
    llm_model_name: str = "Qwen/Qwen2.5-3B"
    batch_size: int = 32
    max_length: int = 2048
    num_samples: int = 10000
    start_point: int = 2000
    output_dir: str = "data/msmarco/"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LLMSoftLabeler:
    """Handles soft labeling of MS MARCO data using an LLM."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.tokenizer = None
        
    def load_model(self) -> None:
        """Load and initialize the LLM model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.config.llm_model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model_name
            ).to(self.device).eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _create_llm_input(self, query: str, passage: str, answer: str) -> str:
        """Create formatted input text for the LLM."""
        return f"Question: {query} Context: {passage} Answer: {answer}"
    
    def _compute_target_mask(self, text: str, batch_idx: int, labels: torch.Tensor) -> None:
        """Mask everything before 'Answer:' in the labels tensor."""
        target_start_idx = text.rfind("Answer: ") + len("Answer: ")
        target_tokens = self.tokenizer(
            text[:target_start_idx], 
            return_tensors="pt"
        )["input_ids"]
        target_start_token_idx = target_tokens.shape[1] - 1
        labels[batch_idx, :target_start_token_idx] = -100
    
    def _compute_batch_losses(self, batch_inputs: List[str]) -> List[float]:
        """Compute losses for a batch of inputs."""
        # Tokenize batch
        inputs = self.tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        labels = inputs["input_ids"].clone()
        
        # Mask everything before "Answer:" for each example
        for k, text in enumerate(batch_inputs):
            self._compute_target_mask(text, k, labels)
        
        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        batch_losses = []
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            logits = outputs.logits
            
            # Compute per-example losses
            for k in range(logits.shape[0]):
                example_logits = logits[k, :-1, :].contiguous()
                example_labels = labels[k, 1:].contiguous()
                valid_positions = example_labels != -100
                
                if valid_positions.sum() > 0:
                    loss_val = F.cross_entropy(
                        example_logits[valid_positions],
                        example_labels[valid_positions]
                    )
                    batch_losses.append(loss_val.item())
                else:
                    batch_losses.append(float('inf'))
        
        return batch_losses
    
    def compute_soft_labels_for_item(self, item: Dict) -> Optional[Dict]:
        """Compute soft labels for a single data item."""
        query = item["query"]
        answers = item["answers"]
        passages = item["passages"]["passage_text"]
        is_selected = item["passages"]["is_selected"]
        
        # Skip if no answer
        if not answers:
            return None
        
        answer = answers[0]
        
        # Add empty passage as negative example
        all_passages = passages + [""]
        all_labels = is_selected + [0]
        
        # Create LLM inputs for all passages
        llm_inputs = [
            self._create_llm_input(query, passage, answer)
            for passage in all_passages
        ]
        
        # Process through LLM in batches
        passage_losses = []
        
        for j in range(0, len(llm_inputs), self.config.batch_size):
            batch_inputs = llm_inputs[j:j + self.config.batch_size]
            batch_losses = self._compute_batch_losses(batch_inputs)
            passage_losses.extend(batch_losses)
        
        # Create new item with soft labels
        return {
            "query": query,
            "answers": answers,
            "passages": {
                "passage_text": all_passages,
                "is_selected": all_labels,
                "soft_labels": passage_losses
            },
            "query_id": item.get("query_id", 0),
            "query_type": item.get("query_type", ""),
            "wellFormedAnswers": item.get("wellFormedAnswers", [])
        }
    
    def process_dataset(self, dataset: Dataset, max_samples: int) -> List[Dict]:
        """Process dataset to add soft labels."""
        logger.info(f"Processing {min(len(dataset), max_samples)} samples")
        
        soft_labeled_data = []
        processed_count = 0
        
        for item in tqdm(dataset, desc="Computing soft labels"):
            if processed_count >= max_samples:
                break
                
            soft_labeled_item = self.compute_soft_labels_for_item(item)
            if soft_labeled_item is not None:
                soft_labeled_data.append(soft_labeled_item)
                processed_count += 1
        
        logger.info(f"Successfully processed {len(soft_labeled_data)} items")
        return soft_labeled_data


def load_msmarco_dataset(start_point: int, num_samples: int) -> Dataset:
    """Load and select subset of MS MARCO dataset."""
    logger.info("Loading MS MARCO dataset")
    try:
        msmarco = load_dataset("microsoft/ms_marco", "v1.1")
        train_data = msmarco["train"]
        selected_data = train_data.select(range(start_point, start_point + num_samples))
        logger.info(f"Selected {len(selected_data)} samples from MS MARCO")
        return selected_data
    except Exception as e:
        logger.error(f"Failed to load MS MARCO dataset: {e}")
        raise


def save_soft_labeled_data(data: List[Dict], config: Config) -> str:
    """Save soft-labeled data to JSON file."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = config.llm_model_name.replace("/", "_")
    filename = f"soft_labels_msmarco_{model_name}_{config.start_point}-{config.start_point + config.num_samples}.json"
    filepath = output_dir / filename
    
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved soft-labeled dataset to {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        raise


def analyze_label_mismatches(data: List[Dict]) -> Tuple[int, int]:
    """Analyze mismatches between hard and soft labels."""
    logger.info("Analyzing hard vs soft label mismatches")
    
    mismatches = 0
    total_examples = 0
    
    for example in data:
        if not example['answers']:  # Skip unanswerable
            continue
        
        hard_labels = example['passages']['is_selected']
        soft_labels = example['passages']['soft_labels']
        
        # Find human-labeled positive (excluding the added empty passage)
        human_positive_idx = None
        for i, label in enumerate(hard_labels[:-1]):  # Exclude last (empty) passage
            if label == 1:
                human_positive_idx = i
                break
        
        if human_positive_idx is not None:
            # Find LLM's best (lowest loss) among original passages
            original_soft_labels = soft_labels[:-1]  # Exclude empty passage
            llm_best_idx = original_soft_labels.index(min(original_soft_labels))
            
            if human_positive_idx != llm_best_idx:
                mismatches += 1
            
            total_examples += 1
    
    mismatch_rate = mismatches / total_examples if total_examples > 0 else 0
    logger.info(f"Mismatch rate: {mismatches}/{total_examples} = {mismatch_rate:.1%}")
    
    return mismatches, total_examples


def main():
    """Main execution function."""
    config = Config()
    
    try:
        # Initialize soft labeler
        labeler = LLMSoftLabeler(config)
        labeler.load_model()
        
        # Load dataset
        dataset = load_msmarco_dataset(config.start_point, config.num_samples)
        
        # Process dataset
        soft_labeled_data = labeler.process_dataset(dataset, config.num_samples)
        
        # Save results
        output_path = save_soft_labeled_data(soft_labeled_data, config)
        
        # Analyze results
        mismatches, total = analyze_label_mismatches(soft_labeled_data)
        
        logger.info("Processing completed successfully")
        logger.info(f"Output saved to: {output_path}")
        logger.info(f"Label mismatches: {mismatches}/{total}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
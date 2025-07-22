"""
Evaluation functions for bitter-retrieval training pipeline.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import Config
from .data import convert_soft_to_hard
from .utils import encode_texts, generate_answer, compute_f1_score, compute_exact_match, compute_llm_loss

logger = logging.getLogger(__name__)


class Evaluator:
    """Main evaluator class for model evaluation."""
    
    def __init__(self, config: Config, encoder_tokenizer, llm_model, llm_tokenizer):
        self.config = config
        self.encoder_tokenizer = encoder_tokenizer
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.device = torch.device(config.device)
    
    def evaluate_ranking(
        self, 
        model, 
        eval_data: List[Dict[str, Any]], 
        label_type: str = "original"
    ) -> Dict[str, float]:
        """
        Evaluate ranking performance on MS MARCO data.
        
        Args:
            model: Encoder model to evaluate
            eval_data: Evaluation dataset
            label_type: Type of labels to use ("original" or "converted")
            
        Returns:
            Dictionary with ranking metrics (MRR, Recall@1, Recall@5)
        """
        logger.info(f"Evaluating ranking performance (label_type={label_type})")
        model.eval()
        ranks = []

        with torch.no_grad():
            for item in tqdm(eval_data, desc=f"Ranking eval ({label_type})"):
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

                # Encode and compute similarities
                query_emb = encode_texts([f"query: {query}"], model, self.encoder_tokenizer)
                passage_embs = encode_texts([f"passage: {p}" for p in passages], model, self.encoder_tokenizer)
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
    
    def evaluate_llm_loss(
        self, 
        model, 
        eval_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate average LLM loss on answer tokens.
        
        Args:
            model: Encoder model to evaluate
            eval_data: Evaluation dataset
            
        Returns:
            Dictionary with average LLM loss
        """
        logger.info("Evaluating LLM loss")
        model.eval()
        all_losses = []

        with torch.no_grad():
            for item in tqdm(eval_data, desc="LLM loss eval"):
                if not item['answers']:
                    continue

                query = item['query']
                answer = item['answers'][0]
                passages = item['passages']['passage_text']

                # Encode and find best passage according to encoder
                query_emb = encode_texts([f"query: {query}"], model, self.encoder_tokenizer)
                passage_embs = encode_texts([f"passage: {p}" for p in passages], model, self.encoder_tokenizer)

                similarities = torch.matmul(query_emb, passage_embs.T).squeeze(0)
                best_passage_idx = similarities.argmax().item()
                best_passage = passages[best_passage_idx]

                # Compute LLM loss for best passage
                loss = compute_llm_loss(
                    query, best_passage, answer,
                    self.llm_model, self.llm_tokenizer,
                    self.device
                )
                
                if loss != float('inf'):
                    all_losses.append(loss)

        return {"Avg_LLM_Loss": np.mean(all_losses) if all_losses else float('inf')}
    
    def evaluate_generation(
        self, 
        model, 
        test_data: List[Dict[str, Any]], 
        num_examples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate answer generation quality.
        
        Args:
            model: Encoder model to evaluate
            test_data: Test dataset
            num_examples: Number of examples to evaluate (None for all)
            
        Returns:
            Dictionary with generation metrics (Exact_Match, F1_Score)
        """
        if num_examples is None:
            num_examples = len(test_data)
        
        logger.info(f"Evaluating generation quality on {num_examples} examples")
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

                # Find best passage using encoder
                query_emb = encode_texts([f"query: {query}"], model, self.encoder_tokenizer)
                passage_embs = encode_texts([f"passage: {p}" for p in passages], model, self.encoder_tokenizer)
                similarities = torch.matmul(query_emb, passage_embs.T).squeeze(0)
                best_passage = passages[similarities.argmax().item()]

                try:
                    generated_answer = generate_answer(
                        query, best_passage, 
                        self.llm_model, self.llm_tokenizer,
                        max_tokens=self.config.model.generation_max_tokens,
                        device=self.device
                    )
                    
                    em = compute_exact_match(generated_answer, reference_answer)
                    f1 = compute_f1_score(generated_answer, reference_answer)
                    exact_matches.append(em)
                    f1_scores.append(f1)
                except Exception as e:
                    logger.warning(f"Generation failed: {e}")
                    continue

        return {
            "Exact_Match": np.mean(exact_matches) if exact_matches else 0.0,
            "F1_Score": np.mean(f1_scores) if f1_scores else 0.0
        }


class SquadEvaluator:
    """Evaluator for SQuAD dataset."""
    
    def __init__(self, config: Config, encoder_tokenizer, llm_model, llm_tokenizer):
        self.config = config
        self.encoder_tokenizer = encoder_tokenizer
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.device = torch.device(config.device)
    
    def precompute_embeddings(self, corpus_texts: List[str], model) -> torch.Tensor:
        """
        Precompute embeddings for SQuAD corpus.
        
        Args:
            corpus_texts: List of context passages
            model: Encoder model
            
        Returns:
            Tensor of precomputed embeddings
        """
        logger.info("Precomputing SQuAD corpus embeddings")
        embeddings = []
        model.eval()

        with torch.no_grad():
            for text in tqdm(corpus_texts, desc="Encoding corpus"):
                emb = encode_texts([f"passage: {text}"], model, self.encoder_tokenizer)
                embeddings.append(emb.cpu())

        return torch.cat(embeddings, dim=0).to(self.device)
    
    def evaluate_squad(
        self,
        model,
        qa_data: List[Dict[str, Any]],
        corpus_embeddings: torch.Tensor,
        corpus_texts: List[str],
        num_examples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate retrieval and generation on SQuAD.
        
        Args:
            model: Encoder model to evaluate
            qa_data: SQuAD QA data
            corpus_embeddings: Precomputed corpus embeddings
            corpus_texts: Corpus text passages
            num_examples: Number of examples to evaluate
            
        Returns:
            Dictionary with SQuAD evaluation metrics
        """
        if num_examples is None:
            num_examples = len(qa_data)
        
        logger.info(f"Evaluating SQuAD performance on {num_examples} examples")
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
                question_emb = encode_texts([f"query: {question}"], model, self.encoder_tokenizer)
                similarities = torch.matmul(question_emb, corpus_embeddings.T).squeeze(0)
                best_context_idx = similarities.argmax().item()
                best_context = corpus_texts[best_context_idx]

                # Check if we retrieved the correct context
                retrieval_hit = (best_context_idx == correct_context_idx)
                retrieval_hits.append(retrieval_hit)

                # Compute LLM loss with retrieved context
                loss = compute_llm_loss(
                    question, best_context, correct_answer,
                    self.llm_model, self.llm_tokenizer,
                    self.device
                )
                
                if loss != float('inf'):
                    llm_losses.append(loss)

                # Generate answer and compute EM/F1
                try:
                    generated_answer = generate_answer(
                        question, best_context,
                        self.llm_model, self.llm_tokenizer,
                        max_tokens=self.config.model.generation_max_tokens,
                        device=self.device
                    )
                    
                    em = compute_exact_match(generated_answer, correct_answer)
                    f1 = compute_f1_score(generated_answer, correct_answer)
                    exact_matches.append(em)
                    f1_scores.append(f1)
                except Exception as e:
                    logger.warning(f"Generation failed: {e}")
                    continue

        return {
            "Retrieval_Accuracy": np.mean(retrieval_hits),
            "LLM_Loss": np.mean(llm_losses) if llm_losses else float('inf'),
            "Exact_Match": np.mean(exact_matches) if exact_matches else 0.0,
            "F1_Score": np.mean(f1_scores) if f1_scores else 0.0,
            "Num_Examples": len(retrieval_hits)
        }


def create_evaluators(config: Config, encoder_tokenizer, llm_model, llm_tokenizer) -> Tuple[Evaluator, SquadEvaluator]:
    """
    Create evaluator instances.
    
    Args:
        config: Configuration object
        encoder_tokenizer: Tokenizer for encoder model
        llm_model: Language model for evaluation
        llm_tokenizer: Tokenizer for language model
        
    Returns:
        Tuple of (Evaluator, SquadEvaluator)
    """
    evaluator = Evaluator(config, encoder_tokenizer, llm_model, llm_tokenizer)
    squad_evaluator = SquadEvaluator(config, encoder_tokenizer, llm_model, llm_tokenizer)
    
    return evaluator, squad_evaluator 
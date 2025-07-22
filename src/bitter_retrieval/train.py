"""
Main training script for bitter-retrieval model.
"""

import logging
import os
from typing import Optional

import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import parse_args, load_config
from .data import set_random_seed, load_train_data, load_eval_data, load_squad_data, get_squad_test_split
from .models import train_model, create_baseline_model
from .utils import setup_logging


def setup_wandb(config: dict) -> Optional[object]:
    """Initialize Weights & Biases logging."""
    if not config["use_wandb"]:
        return None
    
    try:
        run_name = config["run_name"] or f"{config['method']}-temp:{config['temperature']}"
        wandb.init(
            project=config["wandb_project"],
            name=run_name,
            config=config
        )
        return wandb
    except Exception as e:
        logging.error(f"Failed to initialize wandb: {e}")
        return None


def load_models(config: dict):
    """Load encoder tokenizer and LLM for evaluation."""
    logger = logging.getLogger(__name__)
    
    # Load encoder tokenizer
    logger.info(f"Loading encoder tokenizer: {config['encoder_model']}")
    encoder_tokenizer = AutoTokenizer.from_pretrained(
        config["encoder_model"], 
        trust_remote_code=True
    )
    
    # Load LLM for evaluation
    logger.info(f"Loading LLM: {config['llm_model']}")
    llm_model = AutoModelForCausalLM.from_pretrained(
        config["llm_model"]
    ).to(torch.device(config["device"])).eval()
    
    llm_tokenizer = AutoTokenizer.from_pretrained(config["llm_model"])
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    
    return encoder_tokenizer, llm_model, llm_tokenizer


def run_training_experiment(config: dict) -> dict:
    """Run a single training experiment."""
    logger = logging.getLogger(__name__)
    
    # Set random seed
    set_random_seed(config["seed"])
    
    # Setup wandb
    wandb_logger = setup_wandb(config)
    
    try:
        # Load data
        logger.info("Loading datasets...")
        train_data = load_train_data(config)
        eval_data = load_eval_data(config)
        squad_qa_data, squad_corpus, squad_title_contexts = load_squad_data(config)
        squad_test_data = get_squad_test_split(squad_qa_data, config)
        
        logger.info(f"Train examples: {len(train_data)}")
        logger.info(f"Eval examples: {len(eval_data)}")
        logger.info(f"SQuAD examples: {len(squad_qa_data)}")
        
        # Load models
        encoder_tokenizer, llm_model, llm_tokenizer = load_models(config)
        
        # Train model
        logger.info(f"Starting training with method: {config['method']}")
        trained_model = train_model(
            config, 
            train_data, 
            encoder_tokenizer,
            wandb_logger=wandb_logger
        )
        
        logger.info("Training completed successfully")
        
        # Simple final evaluation - just log completion
        final_results = {"status": "completed", "method": config["method"]}
        
        if wandb_logger:
            wandb_logger.log({"final/" + k: v for k, v in final_results.items()})
        
        return final_results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if wandb_logger:
            wandb.finish()


def main():
    """Main execution function."""
    # Setup logging first
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    try:
        # Create config from command line arguments
        config = parse_args()
        
        logger.info("Starting bitter-retrieval training")
        logger.info(f"Configuration: {config}")
        
        # Run training experiment
        results = run_training_experiment(config)
        
        logger.info("Training completed successfully")
        logger.info(f"Results: {results}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Direct experiment runner matching the working research code pattern.
"""

import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.bitter_retrieval.auth import setup_authentication
from src.bitter_retrieval.config import load_config
from src.bitter_retrieval.data import set_random_seed, load_train_data, load_squad_data
from src.bitter_retrieval.models import (
    train_standard_infonce, 
    train_converted_infonce, 
    train_kl_soft_infonce,
    precompute_squad_embeddings,
    evaluate_squad_full
)


def main():
    """Run experiments matching the working research pattern."""
    
    # Load config
    config = load_config()
    
    # Setup authentication
    auth_status = setup_authentication()
    if not auth_status["huggingface"]:
        raise RuntimeError("Hugging Face authentication required")
    
    # Set random seed
    set_random_seed(config["seed"])
    
    # Load data
    print("Loading datasets...")
    train_data = load_train_data(config)
    squad_qa_data, squad_corpus, _ = load_squad_data(config)
    
    # Load models
    print(f"Loading encoder tokenizer: {config['encoder_model']}")
    encoder_tokenizer = AutoTokenizer.from_pretrained(
        config["encoder_model"], 
        trust_remote_code=True
    )
    
    print(f"Loading LLM: {config['llm_model']}")
    device = torch.device(config["device"])
    llm_model = AutoModelForCausalLM.from_pretrained(
        config["llm_model"]
    ).to(device).eval()
    
    llm_tokenizer = AutoTokenizer.from_pretrained(config["llm_model"])
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    
    # Experiment parameters (matching your working code)
    teacher_temps = [.01]
    student_temps = [.01]
    squad_test_set = squad_qa_data[100:500]
    
    results = {}
    
    for s in student_temps:
        for t in teacher_temps:
            # Set run name (matching your pattern)
            run_name = "standard_infonce-inbatchneg-NOMIC-25k-emptystring:include-temp:.02-epochs:2-bs:2-gradacc:8-LR:2e-5-warmup:400"
            
            # Initialize wandb
            wandb.init(
                project="bitter-retrieval",
                entity="nickcdryan",  # Using your entity
                name=run_name,
                notes="Reproduced from working research code",
                config=config
            )
            
            print(f"Starting training: {run_name}")
            
            # Train model (matching your direct call pattern)
            model = train_standard_infonce(
                config=config,
                train_data=train_data,
                encoder_tokenizer=encoder_tokenizer,
                evaluator=None,
                wandb_logger=wandb,
                llm_model=llm_model,
                llm_tokenizer=llm_tokenizer
            )
            
            # Uncomment for other methods:
            # model = train_converted_infonce(
            #     config=config, train_data=train_data, encoder_tokenizer=encoder_tokenizer,
            #     evaluator=None, wandb_logger=wandb, llm_model=llm_model, llm_tokenizer=llm_tokenizer
            # )
            # 
            # model = train_kl_soft_infonce(
            #     config=config, train_data=train_data, encoder_tokenizer=encoder_tokenizer,
            #     evaluator=None, wandb_logger=wandb, llm_model=llm_model, llm_tokenizer=llm_tokenizer
            # )
            
            # Final evaluation on test set
            print("Running final evaluation...")
            squad_embeddings = precompute_squad_embeddings(squad_corpus, model, encoder_tokenizer)
            squad_results = evaluate_squad_full(
                model, encoder_tokenizer, squad_test_set, squad_embeddings, squad_corpus, llm_model, llm_tokenizer
            )
            
            print("Final results:", squad_results)
            results[f"teacher_{t}_student_{s}"] = squad_results
            
            # Log final results to wandb
            wandb.log({
                "final/SQuAD_F1": squad_results['F1_Score'],
                "final/SQuAD_EM": squad_results['Exact_Match'],
                "final/SQuAD_Retrieval": squad_results['Retrieval_Accuracy'],
                "final/SQuAD_Loss": squad_results['LLM_Loss'],
            })
            
            # Cleanup
            wandb.finish()
            del model, squad_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print("All experiments completed:")
    print(results)


if __name__ == "__main__":
    main() 
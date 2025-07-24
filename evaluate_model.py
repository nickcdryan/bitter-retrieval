#!/usr/bin/env python3

import sys
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import train_retrieval
from train_retrieval import (
    get_config, setup_device_and_models, preprocess_squad_title_diverse,
    preprocess_msmarco_validation, precompute_squad_embeddings, evaluate_squad
)
from datasets import load_dataset
import random
import wandb
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate_model.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    config = get_config()
    
    # Setup wandb
    model_name = os.path.basename(model_path)
    if config["wandb_key"]:
        wandb.login(key=config["wandb_key"])
    wandb.init(project=config["wandb_project"], name=f"eval-{model_name}")
    
    # Setup
    device, bert_tokenizer, llm, llm_tokenizer = setup_device_and_models(config)
    train_retrieval.device = device  # Set device in train_retrieval module
    
    # Load saved model
    print(f"Loading model from {model_path}")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    
    # Recreate test datasets with same seed
    random.seed(42)
    squad_dataset = load_dataset("squad_v2")
    squad_qa_data, squad_corpus, _ = preprocess_squad_title_diverse(
        squad_dataset["train"], num_titles=config["squad_num_titles"], 
        questions_per_title=config["squad_questions_per_title"]
    )
    
    msmarco_test_qa_data, msmarco_test_corpus = preprocess_msmarco_validation(
        config, config["msmarco_test_examples"]
    )
    
    # Move LLM to GPU for evaluation
    llm.to(device)
    
    try:
        # SQuAD evaluation
        squad_test_start = config["squad_eval_examples"]
        squad_test_end = squad_test_start + config["squad_test_examples"]
        squad_test_set = squad_qa_data[squad_test_start:squad_test_end]
        
        squad_embeddings = precompute_squad_embeddings(squad_corpus, model, bert_tokenizer, config)
        squad_results = evaluate_squad(model, squad_test_set, squad_embeddings, squad_corpus, 
                                     llm, llm_tokenizer, bert_tokenizer, config)
        
        # MS MARCO evaluation
        msmarco_embeddings = precompute_squad_embeddings(msmarco_test_corpus, model, bert_tokenizer, config)
        msmarco_results = evaluate_squad(model, msmarco_test_qa_data, msmarco_embeddings, 
                                       msmarco_test_corpus, llm, llm_tokenizer, bert_tokenizer, config)
        
    finally:
        llm.cpu()
        torch.cuda.empty_cache()
    
    # Log to wandb
    wandb.log({
        "Final_SQuAD_Retrieval": squad_results['Retrieval_Accuracy'],
        "Final_SQuAD_F1": squad_results['F1_Score'],
        "Final_SQuAD_EM": squad_results['Exact_Match'],
        "Final_SQuAD_LLM_Judge": squad_results['LLM_Judge'],
        "Final_MSMARCO_Retrieval": msmarco_results['Retrieval_Accuracy'],
        "Final_MSMARCO_F1": msmarco_results['F1_Score'],
        "Final_MSMARCO_EM": msmarco_results['Exact_Match'],
        "Final_MSMARCO_LLM_Judge": msmarco_results['LLM_Judge'],
    })
    
    # Results
    print("\n=== FINAL EVALUATION RESULTS ===")
    print(f"SQuAD: Retrieval={squad_results['Retrieval_Accuracy']:.3f}, "
          f"F1={squad_results['F1_Score']:.3f}, EM={squad_results['Exact_Match']:.3f}, "
          f"LLM_Judge={squad_results['LLM_Judge']:.3f}")
    print(f"MSMARCO: Retrieval={msmarco_results['Retrieval_Accuracy']:.3f}, "
          f"F1={msmarco_results['F1_Score']:.3f}, EM={msmarco_results['Exact_Match']:.3f}, "
          f"LLM_Judge={msmarco_results['LLM_Judge']:.3f}")

if __name__ == "__main__":
    main() 
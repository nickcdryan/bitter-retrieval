#!/usr/bin/env python3
"""
Main training script for bitter retrieval models

Usage:
    python scripts/train.py --experiment kl_margin
    python scripts/train.py --config configs/experiments/kl_margin.yaml
    python scripts/train.py --config configs/experiments/full_modular.yaml
    python scripts/train.py  # uses default config
"""

import argparse
import sys
import os
import wandb

# Add the parent directory to the path so we can import bitter_retrieval
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitter_retrieval.config import get_default_config, get_experiment_config, validate_config, load_config_from_yaml
from bitter_retrieval.utils.device import setup_device_and_models
from bitter_retrieval.utils.logging import setup_wandb, log_artifact
from bitter_retrieval.utils.io import save_model
from bitter_retrieval.data.loaders import load_squad_data, load_msmarco_data, load_soft_labeled_data
from bitter_retrieval.training.trainer import ModularTrainer
from bitter_retrieval.evaluation.evaluator import run_validation
from bitter_retrieval.models.encoder import precompute_embeddings
from bitter_retrieval.evaluation.evaluator import evaluate_retrieval

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("python-dotenv not installed - using system environment variables only")


def main():
    parser = argparse.ArgumentParser(description="Train retrieval models with modular loss functions")
    parser.add_argument("--experiment", type=str, help="Experiment name (kl_only, margin_only, kl_margin, etc.)")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--run-name", type=str, help="Custom run name for wandb")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--num-epochs", type=int, help="Override number of epochs")
    
    args = parser.parse_args()
    
    # Get configuration
    if args.config:
        print(f"Loading YAML config: {args.config}")
        config = load_config_from_yaml(args.config)
    elif args.experiment:
        print(f"Using experiment config: {args.experiment}")
        config = get_experiment_config(args.experiment)
    else:
        print("Using default config")
        config = get_default_config()
    
    # Override config with command line arguments
    if args.run_name:
        config["run_name"] = args.run_name
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.num_epochs:
        config["num_epochs"] = args.num_epochs
    
    # Validate configuration
    validate_config(config)
    
    print(f"🚀 Starting training with run name: {config['run_name']}")
    print(f"📊 Training method: {config['training_method']}")
    if config['training_method'] == 'modular':
        print(f"🔧 Loss components: {config['loss_components']}")
    
    # Setup device and models
    device, bert_tokenizer, llm, llm_tokenizer = setup_device_and_models(config)
    print(f"🖥️  Using device: {device}")
    
    # Setup wandb
    setup_wandb(config)
    log_artifact(__file__, "training_script", "code")
    
    # Load data
    print("📂 Loading datasets...")
    
    # Load SQuAD data for evaluation
    squad_qa_data, squad_corpus, title_contexts = load_squad_data(
        num_titles=config["squad_num_titles"], 
        questions_per_title=config["squad_questions_per_title"]
    )
    
    # Load MS MARCO validation data
    msmarco_qa_data, msmarco_corpus = load_msmarco_data(config, config["msmarco_val_examples"])
    
    # Load training data
    soft_train_data = load_soft_labeled_data(config)
    
    # Setup trainer
    trainer = ModularTrainer(config, device)
    trainer.setup_training(bert_tokenizer)
    
    # Create validation function
    def validation_fn(model, step):
        val_results = run_validation(
            model, squad_qa_data, squad_corpus, msmarco_qa_data, msmarco_corpus,
            llm, llm_tokenizer, bert_tokenizer, config, device
        )
        
        # Log validation metrics
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
        wandb.log(val_metrics, step=step)
    
    # Train model
    print("🎯 Starting training...")
    trained_model = trainer.train(soft_train_data, bert_tokenizer, validation_fn)
    
    # Final evaluation
    print("🔍 Running final evaluation...")
    
    # Move LLM to GPU for final evaluation
    llm.to(device)
    
    try:
        # Final SQuAD test
        squad_test_start = config["squad_eval_examples"]
        squad_test_end = squad_test_start + config["squad_test_examples"]
        squad_test_set = squad_qa_data[squad_test_start:squad_test_end]
        
        squad_embeddings = precompute_embeddings(squad_corpus, trained_model, bert_tokenizer, config, "SQuAD Final")
        squad_results = evaluate_retrieval(
            trained_model, squad_test_set, squad_embeddings, squad_corpus, 
            llm, llm_tokenizer, bert_tokenizer, config
        )
        
        # Final MS MARCO test
        from bitter_retrieval.data.loaders import preprocess_msmarco_validation
        from datasets import load_dataset
        ms_marco_dataset = load_dataset(config["dataset_name"])
        msmarco_test_data = list(ms_marco_dataset["validation"])
        msmarco_test_qa_data, msmarco_test_corpus = preprocess_msmarco_validation(
            config, msmarco_test_data, config["msmarco_test_examples"]
        )
        
        msmarco_embeddings = precompute_embeddings(msmarco_test_corpus, trained_model, bert_tokenizer, config, "MS MARCO Final")
        msmarco_results = evaluate_retrieval(
            trained_model, msmarco_test_qa_data, msmarco_embeddings, msmarco_test_corpus,
            llm, llm_tokenizer, bert_tokenizer, config
        )
        
    finally:
        # Move LLM back to CPU and clear cache
        llm.cpu()
        import torch
        torch.cuda.empty_cache()
    
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
    
    print("📈 Final SQuAD results:", squad_results)
    print("📈 Final MS MARCO results:", msmarco_results)
    
    # Save model
    save_model(trained_model, config)
    
    print("✅ Training completed successfully!")


if __name__ == "__main__":
    main() 
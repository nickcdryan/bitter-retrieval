#!/usr/bin/env python3
"""
Model evaluation script for bitter retrieval models

This script evaluates trained retrieval models on SQuAD and MS MARCO datasets.

Usage:
    python scripts/evaluate.py --model-path models/my-model
    python scripts/evaluate.py --model-path models/my-model --dataset squad --num-examples 1000
    python scripts/evaluate.py --encoder nomic-ai/nomic-embed-text-v1 --llm Qwen/Qwen3-8B-Base
"""

import argparse
import sys
import os
import torch

# Add the parent directory to the path so we can import bitter_retrieval
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitter_retrieval.config import get_default_config
from bitter_retrieval.utils.device import setup_device_and_models
from bitter_retrieval.data.loaders import load_squad_data, load_msmarco_data
from bitter_retrieval.models.encoder import precompute_embeddings
from bitter_retrieval.evaluation.evaluator import evaluate_retrieval
from transformers import AutoModel

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("python-dotenv not installed - using system environment variables only")


def load_trained_model(model_path, config, device):
    """Load a trained model from a saved path"""
    print(f"🤖 Loading trained model from: {model_path}")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval models")
    parser.add_argument("--model-path", type=str, help="Path to saved model directory")
    parser.add_argument("--encoder", type=str, help="Encoder model name (if not using saved model)")
    parser.add_argument("--llm", type=str, help="LLM model name")
    parser.add_argument("--dataset", type=str, choices=["squad", "msmarco", "both"], default="both", help="Dataset to evaluate on")
    parser.add_argument("--num-examples", type=int, help="Number of examples to evaluate")
    parser.add_argument("--output-file", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    # Get base configuration
    config = get_default_config()
    
    # Override config based on arguments
    if args.llm:
        config["llm_model"] = args.llm
    if args.encoder:
        config["encoder_model"] = args.encoder
    
    # Setup device and models
    device, bert_tokenizer, llm, llm_tokenizer = setup_device_and_models(config)
    print(f"🖥️  Using device: {device}")
    
    # Load encoder model
    if args.model_path:
        encoder_model = load_trained_model(args.model_path, config, device)
    else:
        from bitter_retrieval.utils.device import setup_encoder_model
        encoder_model = setup_encoder_model(config, device)
        print(f"🤖 Using base encoder model: {config['encoder_model']}")
    
    results = {}
    
    # Evaluate on SQuAD if requested
    if args.dataset in ["squad", "both"]:
        print("\n📊 Evaluating on SQuAD...")
        
        # Load SQuAD data
        squad_qa_data, squad_corpus, _ = load_squad_data(
            num_titles=config["squad_num_titles"], 
            questions_per_title=config["squad_questions_per_title"]
        )
        
        # Determine number of examples
        num_examples = args.num_examples if args.num_examples else len(squad_qa_data)
        
        # Move LLM to GPU for evaluation
        llm.to(device)
        
        try:
            # Precompute embeddings
            squad_embeddings = precompute_embeddings(squad_corpus, encoder_model, bert_tokenizer, config, "SQuAD")
            
            # Run evaluation
            squad_results = evaluate_retrieval(
                encoder_model, squad_qa_data, squad_embeddings, squad_corpus,
                llm, llm_tokenizer, bert_tokenizer, config, num_examples
            )
            
            results["squad"] = squad_results
            print("📈 SQuAD Results:")
            for metric, value in squad_results.items():
                print(f"  {metric}: {value:.4f}")
            
        finally:
            llm.cpu()
            torch.cuda.empty_cache()
    
    # Evaluate on MS MARCO if requested
    if args.dataset in ["msmarco", "both"]:
        print("\n📊 Evaluating on MS MARCO...")
        
        # Load MS MARCO data
        num_examples = args.num_examples if args.num_examples else config["msmarco_test_examples"]
        msmarco_qa_data, msmarco_corpus = load_msmarco_data(config, num_examples)
        
        # Move LLM to GPU for evaluation
        llm.to(device)
        
        try:
            # Precompute embeddings
            msmarco_embeddings = precompute_embeddings(msmarco_corpus, encoder_model, bert_tokenizer, config, "MS MARCO")
            
            # Run evaluation
            msmarco_results = evaluate_retrieval(
                encoder_model, msmarco_qa_data, msmarco_embeddings, msmarco_corpus,
                llm, llm_tokenizer, bert_tokenizer, config, num_examples
            )
            
            results["msmarco"] = msmarco_results
            print("📈 MS MARCO Results:")
            for metric, value in msmarco_results.items():
                print(f"  {metric}: {value:.4f}")
            
        finally:
            llm.cpu()
            torch.cuda.empty_cache()
    
    # Save results if requested
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {args.output_file}")
    
    # Print summary
    print("\n📝 Evaluation Summary:")
    print("=" * 50)
    
    if "squad" in results:
        squad_r = results["squad"]
        print(f"SQuAD - Retrieval: {squad_r['Retrieval_Accuracy']:.3f}, F1: {squad_r['F1_Score']:.3f}, EM: {squad_r['Exact_Match']:.3f}, LLM Judge: {squad_r['LLM_Judge']:.3f}")
    
    if "msmarco" in results:
        msmarco_r = results["msmarco"]
        print(f"MS MARCO - Retrieval: {msmarco_r['Retrieval_Accuracy']:.3f}, F1: {msmarco_r['F1_Score']:.3f}, EM: {msmarco_r['Exact_Match']:.3f}, LLM Judge: {msmarco_r['LLM_Judge']:.3f}")
    
    print("✅ Evaluation completed!")


if __name__ == "__main__":
    main() 
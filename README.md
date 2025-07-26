# Bitter Retrieval

A modular training framework for retrieval models that trains on real-valued labels derived from downstream LLM performance rather than binary human annotations.

## 🎯 Core Hypothesis

Most retrieval models are created by fine-tuning on standard retrieval datasets like MS MARCO, which contain question-answer pairs with context passages labeled by humans as relevant or irrelevant. This traditional approach relies on binary relevance labels that are often noisy and reflect human biases about what constitutes useful context. But what we actually care about is: **does the retrieved context help a downstream LLM generate better answers?**

This framework addresses this by:

1. **Soft Label Generation**: For each (question, context, answer) triplet used in our retrieval dataset (MS MARCO), we measure how well the context actually helps a frozen decoder LLM generate the correct answer by computing the loss over answer tokens. This loss becomes the "soft" utility labels associated with each context.
2. **Real-Valued Training**: Instead of contrastive loss with binary labels, we fine-tune base encoder models models to match similarity scores against these real-valued "soft" utility labels 
3. **End-to-End Evaluation**: Rather than evaluating our retrieval models on how well they recover the (noisy, possibly biased) labels, we evaluate on whether retrieved documents actually improve LLM answer generation

The goal is to directly optimize for downstream performance rather than proxy metrics, testing whether this approach produces better retrieval models for LLM-assisted question answering.

## 📊 Preliminary Results

### Experimental Setup
- **Training Data**: 80k MS MARCO examples
- **Hardware**: 1x H100, 2 epochs
- **Base Models**: [BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased), [nomic-embed-text-v1-unsupervised](https://huggingface.co/nomic-ai/nomic-embed-text-v1-unsupervised)
- **Evaluation**: MS MARCO test data sample. For each test data example, the question and top retrieved context are passed into a newly initialized decoder model that produces an answer. LLM-as-judge (Gemini 2.0) is used to evaluate the correctness of the answer against the ground truth provided in the original dataset. Retrieval accuracy measures whether the retrieved context matches the context marked as "best" by human annotators in the original MS MARCO dataset.

### Results: Same LLM (Qwen 3-8B for both labeling and evaluation)

| Model | Training Method | Answer Accuracy ↑ | Retrieval Accuracy |
|-------|----------------|-------------------|-------------------|
| **BERT Base** | Standard InfoNCE | 45.8% | 30.4% |
| **BERT Base** | **Soft Labels (Ours)** | **47.0%** ✅ | 29.6% |
| **Nomic Embed** | Standard InfoNCE | 47.6% | 40.0% |
| **Nomic Embed** | **Soft Labels (Ours)** | **47.8%** ✅ | 33.6% |

### Results: Cross-LLM Generalization 

**(Nomic Embed trained with Qwen-generated labels → Llama 3.1-8B evaluation)**

| Training Method | Answer Accuracy ↑ | Retrieval Accuracy |
|----------------|-------------------|-------------------|
| Standard InfoNCE | 33.8% | 38.2% |
| **Soft Labels (Ours)** | **37.6%** ✅ | 33.6% |


**(BERT trained with Llama-generated labels → Qwen 3 8B evaluation)**

| Training Method | Answer Accuracy ↑ | Retrieval Accuracy |
|----------------|-------------------|-------------------|
| Standard InfoNCE | 46.4% | 31.8% |
| **Soft Labels (Ours)** | **47.8%** ✅ | 28.2% |

### Key Findings

1. **Validation of Core Hypothesis**: Models trained on soft labels achieve better downstream LLM performance despite sometimes lower retrieval accuracy on human labels, demonstrating that the human labels are suboptimal.
2. **Cross-LLM Generalization**: Models trained with labels from one LLM (Qwen) generalize well to different LLMs (Llama) during evaluation, often performing even better
3. **Human Label Limitations**: 21.9% disagreement between human annotations and actual LLM utility demonstrates the noise in traditional training data

### Dataset
We've created and published a soft-labeled version of MS MARCO v1.1 with ~100k examples labeled using the approach described above:
- **Dataset**: [nickcdryan/ms_marco_softlabel_Qwen3-8B-Base_bf16](https://huggingface.co/datasets/nickcdryan/ms_marco_softlabel_Qwen3-8B-Base_bf16)
- **Labeling Model**: Qwen/Qwen3-8B-Base (bf16)
- **Key Finding**: 21.9% disagreement rate between human-labeled "best" passages and passages that actually produce lowest LLM loss

*Note: This project is actively under development. Results are preliminary and based on initial experiments.*

## 🚀 Quick Start

### Setup
```bash
git clone https://github.com/nickcdryan/bitter-retrieval.git
cd bitter-retrieval
./setup.sh
# Restart shell with source ~/.bashrc or open new terminal
poetry run python setup_env.py
poetry run python tests/test_setup.py
```

### Configure API Keys
Create a `.env` file with your API tokens:
```bash
cp .env.example .env
# Edit .env with your actual API keys
```

### Train a Model
```bash
# Use pre-defined experiments
poetry run python scripts/train.py --experiment kl_margin
poetry run python scripts/train.py --config configs/experiments/full_modular.yaml

# Custom training with overrides
poetry run python scripts/train.py --config configs/experiments/kl_only.yaml --batch-size 32 --run-name "my-experiment"
```

## 📁 Repository Structure

```
bitter-retrieval/
├── bitter_retrieval/          # 🧠 Core library
│   ├── config.py             # Configuration management
│   ├── training/             # Training components (losses, trainer, schedulers)
│   ├── data/                 # Data loading and processing
│   ├── models/               # Model utilities (LLM, encoder)
│   ├── evaluation/           # Evaluation metrics and orchestration
│   └── utils/                # Utilities (device, encoding, logging, I/O)
├── scripts/                  # 🚀 Main entry points
│   ├── train.py             # Main training script
│   ├── label_data.py        # Data labeling with LLMs
│   ├── evaluate.py          # Model evaluation
│   ├── setup_env.py         # Environment setup
│   └── push_to_hf.py        # HuggingFace Hub operations
├── configs/                  # ⚙️ Configuration files
│   ├── default.yaml         # Base configuration
│   ├── experiments/         # Pre-defined experiments
│   └── models/              # Model-specific settings
├── tests/                    # 🧪 Test suite
└── [setup files, poetry config, etc.]
```

## 🎯 Core Features

### 🔧 Modular Training Framework
- **Multiple Loss Functions**: KL divergence, MSE, margin hinge, InfoNCE variants
- **Flexible Combinations**: Mix and match loss components with custom weights
- **Easy Experimentation**: YAML-based configuration with inheritance support

### 📊 Training Methods
- **Modular Training**: Combine multiple loss components (`kl + margin`, `kl + mse + infonce`, etc.)
- **Standard InfoNCE**: Classic contrastive learning
- **Soft Label Learning**: Knowledge distillation from teacher LLMs
- **Margin Learning**: Contrastive learning with margin constraints

### 🤖 Model Support
- **Encoders**: Nomic Embed, BERT, any HuggingFace encoder
- **Teacher LLMs**: Qwen, Llama, any causal language model
- **Optimizations**: Flash Attention 2, gradient clipping, LR scheduling

### 📈 Comprehensive Evaluation
- **Multiple Metrics**: Retrieval accuracy, F1, EM, LLM judge scores
- **Datasets**: SQuAD, MS MARCO with configurable corpus sizes
- **Async Processing**: Batch generation and parallel LLM evaluation


## 🧪 Example Experiments

### KL + Margin Training
```bash
poetry run python scripts/train.py --config configs/experiments/kl_margin.yaml
```

### Pure Knowledge Distillation
```bash
poetry run python scripts/train.py --config configs/experiments/kl_only.yaml
```

### Multi-Component Training
```bash
poetry run python scripts/train.py --config configs/experiments/full_modular.yaml
```

### Custom Loss Combination
```yaml
# custom_experiment.yaml
base_config: "configs/default.yaml"
method:
  training_method: "modular"
  loss_components:
    kl: 0.4
    mse: 0.3
    margin: 0.3
```

## 🏷️ Data Labeling

Generate soft labels for your own datasets:

```bash
# Label MS MARCO data
poetry run python scripts/label_data.py --model Qwen/Qwen3-8B-Base --split train --num 1000

# Process all splits and upload to HF Hub
poetry run python scripts/label_data.py --model Qwen/Qwen3-8B-Base --all-splits --upload-hf "username/dataset-name"
```

## 📊 Model Evaluation

```bash
# Evaluate trained model
poetry run python scripts/evaluate.py --model-path models/my-trained-model

# Evaluate base model
poetry run python scripts/evaluate.py --encoder nomic-ai/nomic-embed-text-v1 --dataset squad

# Save results
poetry run python scripts/evaluate.py --model-path models/my-model --output-file results.json
```

## 🤗 HuggingFace Integration

```bash
# Push trained model to HF Hub
poetry run python scripts/push_to_hf.py --model models/my-model --repo username/model-name

# Push dataset to HF Hub  
poetry run python scripts/push_to_hf.py --dataset data/my-dataset.json --repo username/dataset-name --type dataset

# List local models
poetry run python scripts/push_to_hf.py --list-local
```

## ⚙️ Configuration System

### Pre-defined Experiments
- `kl_margin.yaml` - KL divergence + margin hinge loss
- `kl_only.yaml` - Pure knowledge distillation
- `margin_only.yaml` - Pure contrastive learning
- `infonce_only.yaml` - Standard InfoNCE baseline
- `full_modular.yaml` - Multi-component training

### Model Configurations
- `nomic_embed.yaml` - Optimized for Nomic Embed Text v1
- `qwen_8b.yaml` - Optimized for Qwen 3-8B Base
- `bert_base.yaml` - BERT Base alternative

See [`configs/README.md`](configs/README.md) for detailed configuration documentation.

## 🛠️ Development

### Environment Setup
```bash
poetry run python setup_env.py --install-flash-attention
```

### Testing
```bash
poetry run python tests/test_setup.py  # Test environment setup
```

### Dependencies
- Python 3.11+
- PyTorch 2.7+ with CUDA support
- Transformers, Datasets, W&B
- Optional: Flash Attention 2 for performance

## 📈 Supported Systems

- **Linux**: Ubuntu/Debian, Fedora/RHEL, Arch/Manjaro
- **GPU**: CUDA 12.8+ recommended
- **Memory**: 16GB+ RAM, 8GB+ VRAM recommended

## 🎓 Citation

If you use this framework in your research, please cite:

```bibtex
@software{bitter_retrieval_2024,
  title = {Bitter Retrieval: Modular Training Framework for Retrieval Models},
  author = {Nick Ryan},
  year = {2024},
  url = {https://github.com/nickcdryan/bitter-retrieval}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


# Configuration Files

This directory contains YAML configuration files for bitter retrieval training experiments.

## Structure

```
configs/
├── default.yaml              # Base configuration with all default settings
├── experiments/              # Pre-defined experiment configurations
│   ├── kl_margin.yaml       # KL + Margin loss combination
│   ├── kl_only.yaml         # Pure KL divergence loss
│   ├── margin_only.yaml     # Pure margin hinge loss
│   ├── infonce_only.yaml    # Standard InfoNCE loss
│   └── full_modular.yaml    # Multi-component loss combination
└── models/                  # Model-specific configurations
    ├── nomic_embed.yaml     # Nomic Embed Text v1 settings
    ├── qwen_8b.yaml         # Qwen 3-8B Base settings
    └── bert_base.yaml       # BERT Base alternative
```

## Usage

### 1. Using Pre-defined Experiments

```bash
# Train with KL + Margin experiment
python scripts/train.py --config configs/experiments/kl_margin.yaml

# Train with InfoNCE only
python scripts/train.py --config configs/experiments/infonce_only.yaml

# Train with full modular approach (multiple loss components)
python scripts/train.py --config configs/experiments/full_modular.yaml
```

### 2. Combining Experiment and Model Configs

You can combine multiple configs by specifying a model config and using inheritance:

```yaml
# custom_experiment.yaml
base_config: "experiments/kl_margin.yaml"

# Override model settings
models:
  encoder_model: "google-bert/bert-base-uncased"

# Override training settings  
training:
  batch_size: 32
  learning_rate: 3.0e-5
```

### 3. Configuration Inheritance

Configs support inheritance using the `base_config` field:

```yaml
# experiment.yaml
base_config: "../default.yaml"  # Inherit from default

# Override specific settings
logging:
  run_name: "my-custom-experiment"

method:
  loss_components:
    kl: 0.7
    margin: 0.3
```

### 4. Command-line Overrides

You can still override config values from the command line:

```bash
# Use YAML config but override specific values
python scripts/train.py \
  --config configs/experiments/kl_margin.yaml \
  --batch-size 32 \
  --learning-rate 1e-5 \
  --run-name "my-experiment"
```

## Configuration Sections

### `logging`
- `wandb_project`: W&B project name
- `run_name`: Experiment run name

### `models`
- `encoder_model`: HuggingFace encoder model name
- `llm_model`: HuggingFace LLM model name

### `data`
- `dataset_name`: Training dataset name
- `num_data_examples`: Number of examples (-1 for all)
- `encode_max_length`: Max sequence length for encoder
- `llm_max_length`: Max sequence length for LLM

### `training`
- `batch_size`: Training batch size
- `learning_rate`: Learning rate
- `num_epochs`: Number of training epochs
- `validation_frequency`: Validation frequency (steps)
- `gradient_clipping`: Enable gradient clipping
- `grad_clip_max_norm`: Max gradient norm

### `method`
- `training_method`: Training method (`modular`, `standard_infonce`, etc.)
- `loss_components`: Loss component weights for modular training

### `loss_params`
- `teacher_temp`: Teacher temperature for KL loss
- `student_temp`: Student temperature for KL loss
- `margin`: Margin for hinge loss
- `infonce_temperature`: Temperature for InfoNCE loss

### `evaluation`
- SQuAD and MS MARCO evaluation parameters

### `saving`
- `save_model`: Whether to save trained models
- `model_save_path`: Path to save models

## Creating Custom Experiments

1. **Copy an existing experiment**: Start with a similar experiment config
2. **Modify loss components**: Adjust weights in `method.loss_components`
3. **Tune hyperparameters**: Modify `loss_params` and `training` sections
4. **Test**: Run training with your custom config

Example custom config:

```yaml
base_config: "../default.yaml"

logging:
  run_name: "kl-mse-experiment"

method:
  training_method: "modular"
  loss_components:
    kl: 0.6
    mse: 0.4

loss_params:
  teacher_temp: 0.005  # Lower temperature
  student_temp: 0.01

training:
  learning_rate: 1.5e-5  # Lower learning rate
  num_epochs: 3
```

## Available Loss Components

For `modular` training method:

- `kl`: KL divergence loss (knowledge distillation)
- `mse`: Mean squared error between similarities
- `margin`: Margin hinge loss (contrastive)
- `standard_infonce`: InfoNCE with original labels
- `converted_infonce`: InfoNCE with converted soft labels

## Environment Variables

Set these in your `.env` file:

```bash
HUGGINGFACE_TOKEN=your_token_here
WANDB_API_KEY=your_wandb_key_here
GEMINI_API_KEY=your_gemini_key_here
``` 
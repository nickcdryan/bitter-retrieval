"""
Modular trainer for retrieval models
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from collections import defaultdict
from typing import Dict, Any, List

from ..utils.device import setup_encoder_model
from ..utils.encoding import encode_texts
from ..data.processors import should_skip_item
from .losses import compute_combined_loss
from .schedulers import create_lr_scheduler


class ModularTrainer:
    """Modular trainer that supports different loss combinations"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_step = 0
        
    def setup_training(self, bert_tokenizer):
        """Setup model, optimizer, and scheduler"""
        # Setup model
        self.model = setup_encoder_model(self.config, self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"])
        
        # Setup scheduler
        # Estimate total steps (will be updated with actual data length)
        self.total_steps = 1000  # placeholder, will update in train()
        
        print(f"Training with method: {self.config['training_method']}")
        if self.config['training_method'] == 'modular':
            print(f"Loss components: {self.config['loss_components']}")
    
    def train_epoch(self, soft_train_data: List[Dict], bert_tokenizer, validation_fn=None):
        """Train one epoch with the configured loss components"""
        self.model.train()
        total_loss = 0
        epoch_loss_components = defaultdict(float)
        
        # Update scheduler with actual data length
        if self.scheduler is None:
            self.total_steps = (len(soft_train_data) // self.config["batch_size"]) * self.config["num_epochs"]
            self.scheduler = create_lr_scheduler(self.optimizer, self.config, self.total_steps)
        
        for i in tqdm(range(0, len(soft_train_data), self.config["batch_size"]), desc="Training"):
            batch_items = soft_train_data[i:i+self.config["batch_size"]]
            batch_items = [item for item in batch_items if not should_skip_item(item)]
            if len(batch_items) == 0:
                continue
            
            # Prepare batch data
            queries = []
            passages = []
            passage_counts = []
            soft_label_groups = []
            hard_label_groups = []
            
            for item in batch_items:
                q = item["query"]
                p_list = item["passages"]["passage_text"]
                l_list = item["passages"]["soft_labels"]
                h_list = item["passages"]["is_selected"]  # original hard labels
                
                queries.append(f"query: {q}")
                passages.extend([f"passage: {p}" for p in p_list])
                soft_label_groups.append(torch.tensor(l_list, dtype=torch.float32, device=self.device))
                hard_label_groups.append(h_list)
                passage_counts.append(len(p_list))
            
            # Batch encoding
            query_embs = encode_texts(queries, self.model, bert_tokenizer, self.config["encode_max_length"], self.device)
            passage_embs = encode_texts(passages, self.model, bert_tokenizer, self.config["encode_max_length"], self.device)
            passage_emb_groups = torch.split(passage_embs, passage_counts, dim=0)
            
            # Compute loss for each item in batch
            batch_loss = 0
            batch_loss_components = defaultdict(float)
            
            for q_emb, p_embs, soft_labels, hard_labels in zip(query_embs, passage_emb_groups, soft_label_groups, hard_label_groups):
                similarities = F.cosine_similarity(q_emb.unsqueeze(0), p_embs, dim=1)
                
                if self.config["training_method"] == "modular":
                    item_loss, item_loss_dict = compute_combined_loss(
                        similarities, soft_labels, hard_labels, self.config, self.config["loss_components"]
                    )
                else:
                    # Handle other training methods here if needed
                    raise NotImplementedError(f"Training method {self.config['training_method']} not implemented in ModularTrainer")
                
                batch_loss += item_loss
                
                # Accumulate loss components
                for key, value in item_loss_dict.items():
                    batch_loss_components[key] += value
            
            # Normalize by batch size
            if len(batch_items) > 0:
                batch_loss = batch_loss / len(batch_items)
                for key in batch_loss_components:
                    batch_loss_components[key] /= len(batch_items)
            
            # Log individual loss components
            log_dict = {"total_loss": batch_loss.item()}
            log_dict.update(batch_loss_components)
            wandb.log(log_dict, step=self.train_step)
            
            # Backward pass
            if batch_loss > 0:
                self.optimizer.zero_grad()
                batch_loss.backward()
                if self.config["gradient_clipping"]:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["grad_clip_max_norm"])
                self.optimizer.step()
                self.scheduler.step()
                total_loss += batch_loss.item()
                
                # Accumulate epoch loss components
                for key, value in batch_loss_components.items():
                    epoch_loss_components[key] += value
            
            # Validation
            if validation_fn and self.train_step % self.config["validation_frequency"] == 0:
                self.model.eval()
                validation_fn(self.model, self.train_step)
                self.model.train()
            
            self.train_step += 1
        
        return total_loss, dict(epoch_loss_components)
    
    def train(self, soft_train_data: List[Dict], bert_tokenizer, validation_fn=None):
        """Full training loop"""
        for epoch in range(self.config["num_epochs"]):
            epoch_loss, epoch_loss_components = self.train_epoch(soft_train_data, bert_tokenizer, validation_fn)
            
            # Print epoch summary
            print(f"Epoch {epoch+1} total loss: {epoch_loss:.4f}")
            for key, value in epoch_loss_components.items():
                print(f"  {key}: {value:.4f}")
        
        return self.model 
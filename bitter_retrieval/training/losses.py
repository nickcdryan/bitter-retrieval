"""
Loss functions for retrieval training
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List
from collections import defaultdict


def compute_kl_loss(student_similarities: torch.Tensor, teacher_soft_labels: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    """KL divergence loss between student and teacher distributions"""
    teacher_probs = F.softmax(-teacher_soft_labels / config["teacher_temp"], dim=0)
    log_student_probs = F.log_softmax(student_similarities / config["student_temp"], dim=0)
    loss = F.kl_div(log_student_probs, teacher_probs, reduction='sum')
    return loss


def compute_mse_loss(student_similarities: torch.Tensor, teacher_soft_labels: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    """MSE loss between student similarities and negative teacher losses"""
    # Convert teacher losses to similarity-like scores (negative and normalized)
    teacher_scores = -teacher_soft_labels
    # Normalize to [0,1] range like cosine similarities
    teacher_scores = (teacher_scores - teacher_scores.min()) / (teacher_scores.max() - teacher_scores.min() + 1e-8)
    
    # MSE between normalized teacher scores and student similarities 
    loss = F.mse_loss(student_similarities, teacher_scores)
    return loss


def compute_margin_loss(student_similarities: torch.Tensor, teacher_soft_labels: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    """Margin hinge loss: positive passage should have higher similarity than negatives by margin"""
    # Best passage has lowest teacher loss
    best_idx = teacher_soft_labels.argmin().item()
    positive_sim = student_similarities[best_idx]
    
    # Get negative similarities (all except the positive one)
    negative_sims = torch.cat([student_similarities[:best_idx], student_similarities[best_idx+1:]])
    
    if len(negative_sims) == 0:
        return torch.tensor(0.0, device=student_similarities.device)
    
    # Hinge loss: max(0, margin - (pos_sim - neg_sim))
    hinge_loss = torch.clamp(config["margin"] - (positive_sim - negative_sims), min=0).mean()
    return hinge_loss


def compute_standard_infonce_loss(student_similarities: torch.Tensor, teacher_hard_labels: List[int], config: Dict[str, Any]) -> torch.Tensor:
    """Standard InfoNCE loss using original hard labels"""
    # Find positive and negative indices from original hard labels
    try:
        pos_idx = teacher_hard_labels.index(1)
    except ValueError:
        # No positive label found, fallback to first passage
        pos_idx = 0
    
    neg_indices = [i for i, label in enumerate(teacher_hard_labels) if label == 0]
    
    if len(neg_indices) == 0:
        return torch.tensor(0.0, device=student_similarities.device)
    
    # Get positive and negative similarities
    pos_sim = student_similarities[pos_idx:pos_idx+1]
    neg_sims = student_similarities[neg_indices]
    
    # Standard InfoNCE loss
    logits = torch.cat([pos_sim, neg_sims]) / config["infonce_temperature"]
    labels = torch.zeros(1, dtype=torch.long, device=student_similarities.device)
    loss = F.cross_entropy(logits.unsqueeze(0), labels)
    return loss


def compute_converted_infonce_loss(student_similarities: torch.Tensor, teacher_soft_labels: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    """InfoNCE loss using converted hard labels from soft labels"""
    # Convert soft labels to hard labels (lowest loss = positive)
    hard_labels = [0] * len(teacher_soft_labels)
    best_idx = teacher_soft_labels.argmin().item()
    hard_labels[best_idx] = 1
    
    # Standard InfoNCE
    pos_sim = student_similarities[best_idx:best_idx+1]
    neg_sims = torch.cat([student_similarities[:best_idx], student_similarities[best_idx+1:]])
    
    if len(neg_sims) == 0:
        return torch.tensor(0.0, device=student_similarities.device)
    
    logits = torch.cat([pos_sim, neg_sims]) / config["infonce_temperature"]
    labels = torch.zeros(1, dtype=torch.long, device=student_similarities.device)
    loss = F.cross_entropy(logits.unsqueeze(0), labels)
    return loss


def compute_combined_loss(student_similarities: torch.Tensor, teacher_soft_labels: torch.Tensor, 
                         teacher_hard_labels: List[int], config: Dict[str, Any], 
                         loss_components: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combine multiple loss components with weights"""
    total_loss = 0.0
    loss_dict = {}
    
    for loss_name, weight in loss_components.items():
        if weight == 0.0:
            continue
            
        if loss_name == "mse":
            component_loss = compute_mse_loss(student_similarities, teacher_soft_labels, config)
        elif loss_name == "kl":
            component_loss = compute_kl_loss(student_similarities, teacher_soft_labels, config)
        elif loss_name == "converted_infonce":
            component_loss = compute_converted_infonce_loss(student_similarities, teacher_soft_labels, config)
        elif loss_name == "standard_infonce":
            component_loss = compute_standard_infonce_loss(student_similarities, teacher_hard_labels, config)
        elif loss_name == "margin":
            component_loss = compute_margin_loss(student_similarities, teacher_soft_labels, config)
        else:
            raise ValueError(f"Unknown loss component: {loss_name}")
        
        weighted_loss = weight * component_loss
        total_loss += weighted_loss
        loss_dict[f"{loss_name}_loss"] = component_loss.item()
        loss_dict[f"{loss_name}_weighted"] = weighted_loss.item()
    
    return total_loss, loss_dict 
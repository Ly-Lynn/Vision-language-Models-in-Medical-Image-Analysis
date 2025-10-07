import pdb
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Any

import pandas as pd
import numpy as np
from sklearn import multiclass
import torch
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from logging_config import get_logger
from tqdm import tqdm

import constants

logger = get_logger(__name__)

class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators
    """
    
    def __init__(
        self,
        model,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            model: Vision-language model to evaluate
            device: Device to run evaluation ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Put model in eval mode
        if hasattr(self.model, 'eval'):
            self.model.eval()
    
    @abstractmethod
    def evaluate(self, dataloader, **kwargs) -> Dict[str, Any]:
        """
        Perform evaluation
        
        Args:
            dataloader: DataLoader test data
            
        Returns:
            Dictionary contains metrics
        """
        pass
    
    def _move_to_device(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Move data to device"""
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(self.device)
        return data


class ZeroShotEvaluator(BaseEvaluator):
    """
    Evaluator for zero-shot classification tasks
    """
    
    def __init__(
        self,
        model,
        class_names: List[str],
        templates: Optional[List[str]] = None,
        mode: str = 'binary',
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            model: Vision-language model (MedCLIP, BioMedCLIP, etc.)
            class_names: List of class names
            templates: Text templates for prompts (if None will use default)
            mode: 'binary' or 'multilabel'
            device: Device to run evaluation
        """
        super().__init__(model, device, **kwargs)
        
        self.class_names = class_names
        self.mode = mode
        
        # Setup default templates
        if templates is None:
            model_name = getattr(model, 'model_name', 'general')
            if model_name in constants.DEFAULT_TEMPLATES:
                base_template = constants.DEFAULT_TEMPLATES[model_name]
                self.templates = [base_template + '{}']
            else:
                self.templates = ['this is a photo of {}']
        else:
            self.templates = templates
        
        # Create text prompts for all classes
        self.text_prompts = self._create_text_prompts()
        
        # Pre-encode text prompts to speed up
        self.text_features = self._encode_text_prompts()
    
    def _create_text_prompts(self) -> List[str]:
        """Create text prompts from class names and templates"""
        prompts = []
        for class_name in self.class_names:
            for template in self.templates:
                prompts.append(template.format(class_name))
        return prompts
    
    def _encode_text_prompts(self) -> torch.Tensor:
        """Pre-encode text prompts"""
        with torch.no_grad():
            if hasattr(self.model, 'encode_text'):
                # For MedCLIP and BioMedCLIP
                text_features = self.model.encode_text(self.text_prompts, normalize=True)
            else:
                logger.error("Model does not have `encode_text` method")
            
            return text_features.to(self.device)
    
    def evaluate(
        self, 
        dataloader, 
        top_k: List[int] = [1, 5],
        return_predictions: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform zero-shot classification evaluation
        
        Args:
            dataloader: DataLoader test data
            top_k: List of k values for top-k accuracy
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary contains accuracy, precision, recall, f1, and optionally predictions
        """
        all_predictions = []
        all_labels = []
        all_logits = []
        
        logger.info(f"🔄 Evaluating zero-shot classification with {len(self.class_names)} classes...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Zero-shot evaluation"):
                batch = self._move_to_device(batch)
                
                # Encode images
                if hasattr(self.model, 'encode_image'):
                    image_features = self.model.encode_image(
                        batch['pixel_values'], normalize=True
                    )
                else:
                    logger.error("Model does not have `encode_image` method")
                
                # Compute similarities with text prompts
                similarities = torch.matmul(image_features, self.text_features.t())
                
                # Aggregate similarities per class if there are multiple templates
                if len(self.templates) > 1:
                    # Reshape: (batch_size, num_classes, num_templates)
                    similarities = similarities.view(
                        similarities.size(0), 
                        len(self.class_names), 
                        len(self.templates)
                    )
                    # Average over templates
                    logits = similarities.mean(dim=-1)
                else:
                    logits = similarities
                
                # Get predictions
                if self.mode == 'multiclass' or self.mode == 'binary':
                    predictions = torch.argmax(logits, dim=-1)
                else:  # multilabel
                    predictions = torch.sigmoid(logits) > 0.5
                
                all_logits.append(logits.cpu())
                all_predictions.append(predictions.cpu())
                all_labels.append(batch['labels'].cpu())
        
        # Concatenate all results
        all_logits = torch.cat(all_logits, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute metrics
        metrics = self._compute_metrics(
            all_predictions, all_labels, all_logits, top_k
        )
        
        if return_predictions:
            metrics['predictions'] = all_predictions.numpy()
            metrics['labels'] = all_labels.numpy()
            metrics['logits'] = all_logits.numpy()
        
        return metrics
    
    def _compute_metrics(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor, 
        logits: torch.Tensor,
        top_k: List[int]
    ) -> Dict[str, Any]:
        """Compute metrics"""
        metrics = {}
        
        predictions_np = predictions.numpy()
        labels_np = labels.numpy()
        logits_np = logits.numpy()
        
        if self.mode == 'binary' or self.mode == 'multiclass':
            # Accuracy
            accuracy = (predictions_np == labels_np).mean()
            metrics['accuracy'] = accuracy
            
            # Top-k accuracy
            for k in top_k:
                if k <= len(self.class_names):
                    top_k_preds = torch.topk(logits, k, dim=-1)[1]
                    top_k_acc = (top_k_preds == labels.unsqueeze(-1)).any(dim=-1).float().mean()
                    metrics[f'top_{k}_accuracy'] = top_k_acc.item()
            
            # Classification report
            try:
                report = classification_report(
                    labels_np, predictions_np, 
                    target_names=self.class_names,
                    output_dict=True,
                    zero_division=0
                )
                metrics['precision'] = report['macro avg']['precision']
                metrics['recall'] = report['macro avg']['recall']
                metrics['f1'] = report['macro avg']['f1-score']
                
                # Per-class metrics
                for i, class_name in enumerate(self.class_names):
                    if str(i) in report:
                        metrics[f'{class_name}_precision'] = report[str(i)]['precision']
                        metrics[f'{class_name}_recall'] = report[str(i)]['recall']
                        metrics[f'{class_name}_f1'] = report[str(i)]['f1-score']
            except Exception as e:
                logger.info(f"⚠️ Cannot compute classification report: {e}")
                metrics['precision'] = 0.0
                metrics['recall'] = 0.0
                metrics['f1'] = 0.0
        
        elif self.mode == 'multilabel':
            # Multilabel metrics
            probabilities = torch.sigmoid(logits).numpy()
            
            # AUC and AUPRC for each class
            auc_scores = []
            auprc_scores = []
            
            for i in range(len(self.class_names)):
                try:
                    auc = roc_auc_score(labels_np[:, i], probabilities[:, i])
                    auprc = average_precision_score(labels_np[:, i], probabilities[:, i])
                    auc_scores.append(auc)
                    auprc_scores.append(auprc)
                except ValueError:
                    # Case where there is only 1 class in labels
                    auc_scores.append(0.0)
                    auprc_scores.append(0.0)
            
            metrics['mean_auc'] = np.mean(auc_scores)
            metrics['mean_auprc'] = np.mean(auprc_scores)
            
            # Per-class AUC
            for i, class_name in enumerate(self.class_names):
                metrics[f'{class_name}_auc'] = auc_scores[i]
                metrics[f'{class_name}_auprc'] = auprc_scores[i]
        
        return metrics


class TextToImageRetrievalEvaluator(BaseEvaluator):
    """
    Evaluator for text-to-image retrieval tasks
    """
    
    def __init__(
        self,
        model,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            model: Vision-language model (MedCLIP, BioMedCLIP, etc.)
            device: Device to run evaluation
        """
        super().__init__(model, device, **kwargs)
    
    def evaluate(
        self,
        image_dataloader,
        text_queries: List[str],
        ground_truth_pairs: List[Tuple[int, int]],  # (text_idx, image_idx)
        top_k_list: List[int] = [1, 5, 10, 20, 50],
        batch_size: int = 64,
        return_rankings: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform text-to-image retrieval evaluation
        
        Args:
            image_dataloader: DataLoader test data
            text_queries: List of text queries
            ground_truth_pairs: List of (text_query_idx, correct_image_idx)
            top_k_list: List of k values for Recall@k
            batch_size: Batch size for text encoding
            return_rankings: Whether to return rankings
            
        Returns:
            Dictionary contains Recall@k, MRR, Mean Rank metrics
        """
        logger.info(f"🔄 Encoding {len(image_dataloader.dataset)} images...")
        
        # Encode all images
        image_embeddings = self._encode_images(image_dataloader)
        
        logger.info(f"🔄 Encoding {len(text_queries)} text queries...")
        
        # Encode all text queries
        text_embeddings = self._encode_texts(text_queries, batch_size)
        
        logger.info(f"🔄 Computing similarities and rankings...")
        
        # Compute similarities
        similarities = cosine_similarity(
            text_embeddings.cpu().numpy(),
            image_embeddings.cpu().numpy()
        )
        
        # Compute rankings for each query
        all_rankings = []
        
        for text_idx, correct_image_idx in tqdm(ground_truth_pairs, desc="Computing rankings"):
            query_similarities = similarities[text_idx]
            
            # Sort by similarity (descending)
            sorted_indices = np.argsort(query_similarities)[::-1]
            
            # Find rank of correct image
            rank_positions = np.where(sorted_indices == correct_image_idx)[0]
            if len(rank_positions) > 0:
                rank = rank_positions[0] + 1  # 1-indexed
            else:
                rank = len(sorted_indices) + 1  # Worst case
            
            all_rankings.append(rank)
        
        # Compute metrics
        metrics = self._compute_retrieval_metrics(all_rankings, top_k_list)
        
        if return_rankings:
            metrics['rankings'] = all_rankings
            metrics['similarities'] = similarities
        
        return metrics
    
    def _encode_images(self, image_dataloader) -> torch.Tensor:
        """Encode all images in dataloader"""
        image_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(image_dataloader, desc="Encoding images"):
                batch = self._move_to_device(batch)
                
                if hasattr(self.model, 'encode_image'):
                    embeddings = self.model.encode_image(
                        batch['pixel_values'], normalize=True
                    )
                else:
                    logger.error("Model does not have `encode_image` method")
                
                image_embeddings.append(embeddings.cpu())
        
        return torch.cat(image_embeddings, dim=0)
    
    def _encode_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        """Encode list of texts"""
        text_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
                batch_texts = texts[i:i + batch_size]
                
                if hasattr(self.model, 'encode_text'):
                    embeddings = self.model.encode_text(batch_texts, normalize=True)
                else:
                    logger.error("Model does not have `encode_text` method")
                
                text_embeddings.append(embeddings.cpu())
        
        return torch.cat(text_embeddings, dim=0)
    
    def _compute_retrieval_metrics(
        self, 
        rankings: List[int], 
        top_k_list: List[int]
    ) -> Dict[str, Any]:
        """Compute retrieval metrics"""
        metrics = {}
        
        # Recall@k
        for k in top_k_list:
            hits = sum(1 for rank in rankings if 0 < rank <= k)
            recall_at_k = hits / len(rankings)
            metrics[f'Recall@{k}'] = recall_at_k
        
        # Mean Reciprocal Rank (MRR)
        reciprocal_ranks = [1.0 / rank if rank > 0 else 0.0 for rank in rankings]
        mrr = np.mean(reciprocal_ranks)
        metrics['MRR'] = mrr
        
        # Mean Rank
        valid_rankings = [rank for rank in rankings if rank > 0]
        if valid_rankings:
            metrics['Mean_Rank'] = np.mean(valid_rankings)
            metrics['Median_Rank'] = np.median(valid_rankings)
        else:
            metrics['Mean_Rank'] = float('inf')
            metrics['Median_Rank'] = float('inf')
        
        # Success rate (percentage of queries that found correct image)
        found_count = sum(1 for rank in rankings if rank > 0)
        metrics['Success_Rate'] = found_count / len(rankings)
        
        return metrics
    
    def evaluate_image_to_text_retrieval(
        self,
        image_dataloader,
        text_queries: List[str], 
        ground_truth_pairs: List[Tuple[int, int]],  # (image_idx, text_idx)
        top_k_list: List[int] = [1, 5, 10, 20, 50],
        batch_size: int = 64,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform image-to-text retrieval evaluation (opposite of text-to-image)
        
        Args:
            image_dataloader: DataLoader test data
            text_queries: List of text queries
            ground_truth_pairs: List of (image_idx, correct_text_idx)
            top_k_list: List of k values for Recall@k
            batch_size: Batch size for text encoding
            
        Returns:
            Dictionary contains Recall@k, MRR, Mean Rank metrics
        """
        logger.info(f"🔄 Image-to-text retrieval evaluation...")
        
        # Encode images and texts
        image_embeddings = self._encode_images(image_dataloader)
        text_embeddings = self._encode_texts(text_queries, batch_size)
        
        # Compute similarities (images x texts)
        similarities = cosine_similarity(
            image_embeddings.cpu().numpy(),
            text_embeddings.cpu().numpy()
        )
        
        # Compute rankings for each image
        all_rankings = []
        
        for image_idx, correct_text_idx in tqdm(ground_truth_pairs, desc="Computing rankings"):
            image_similarities = similarities[image_idx]
            
            # Sort by similarity (descending)
            sorted_indices = np.argsort(image_similarities)[::-1]
            
            # Find rank of correct text
            rank_positions = np.where(sorted_indices == correct_text_idx)[0]
            if len(rank_positions) > 0:
                rank = rank_positions[0] + 1  # 1-indexed
            else:
                rank = len(sorted_indices) + 1  # Worst case
            
            all_rankings.append(rank)
        
        # Compute metrics
        metrics = self._compute_retrieval_metrics(all_rankings, top_k_list)
        
        # Add prefix to distinguish from text-to-image
        prefixed_metrics = {}
        for key, value in metrics.items():
            prefixed_metrics[f'I2T_{key}'] = value
        
        return prefixed_metrics
import os
import torch
import torch.nn as nn
from PIL import Image
import open_clip
from typing import Optional, Dict, List, Union
import numpy as np
from collections import defaultdict


class BioMedCLIPModel(nn.Module):
    """
    BioMedCLIP model implementation using open_clip.
    This model provides text and vision encoding capabilities for medical images.
    """
    
    def __init__(
        self,
        model_name: str = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        context_length: int = 256,
        checkpoint: Optional[str] = None
    ):
        """
        Initialize BioMedCLIP model.
        
        Args:
            model_name: Name/path of the pretrained model from HuggingFace Hub
            context_length: Maximum context length for text tokenization
            checkpoint: Optional checkpoint path to load model weights from
        """
        super().__init__()
        

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        
        # Load model and preprocessing
        self.model, self.preprocess = open_clip.create_model_from_pretrained(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.context_length = context_length
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Load checkpoint if provided
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        
        # Set model to eval mode by default
        self.model.eval()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f'Loaded model weights from: {checkpoint_path}')
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode text inputs to embeddings.
        
        Args:
            texts: Single text string or list of text strings
            normalize: Whether to normalize the embeddings
            
        Returns:
            Text embeddings tensor
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize texts
        text_tokens = self.tokenizer(texts, context_length=self.context_length)
        text_tokens = text_tokens.to(self.device)
        
        # Encode texts
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        
        if normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def encode_image(
        self,
        images: Union[torch.Tensor, List[Image.Image], Image.Image],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode image inputs to embeddings.
        
        Args:
            images: Image tensor, PIL Image, or list of PIL Images
            normalize: Whether to normalize the embeddings
            
        Returns:
            Image embeddings tensor
        """
        # Handle different input types
        if isinstance(images, Image.Image):
            images = [images]
        
        if isinstance(images, list):
            # Process PIL images
            image_tensors = torch.stack([self.preprocess(img) for img in images])
            image_tensors = image_tensors.to(self.device)
        else:
            # Assume tensor input
            image_tensors = images.to(self.device)
        
        # Encode images
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors)
        
        if normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        return_dict: bool = True
    ) -> Union[Dict, tuple]:
        """
        Forward pass for BioMedCLIP model.
        
        Args:
            images: Image tensors (alternative to pixel_values)
            texts: Text strings for tokenization
            pixel_values: Preprocessed image tensors
            input_ids: Tokenized text input ids
            attention_mask: Attention mask for text inputs
            return_loss: Whether to compute and return contrastive loss
            return_dict: Whether to return outputs as dictionary
            
        Returns:
            Dictionary or tuple containing image/text embeddings, logits, and optionally loss
        """
        # Handle image inputs
        if pixel_values is not None:
            images = pixel_values
        
        if images is not None:
            images = images.to(self.device)
            # Handle grayscale images
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
        
        # Handle text inputs
        if texts is not None and input_ids is None:
            text_tokens = self.tokenizer(texts, context_length=self.context_length)
            input_ids = text_tokens
        
        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        
        # Forward pass through the model
        with torch.no_grad():
            if images is not None and input_ids is not None:
                # Both image and text inputs
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(input_ids)
                
                # Get logit scale
                logit_scale = self.model.logit_scale.exp()
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity logits
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
                
            elif images is not None:
                # Only image inputs
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = None
                logits_per_image = None
                logits_per_text = None
                
            elif input_ids is not None:
                # Only text inputs
                text_features = self.model.encode_text(input_ids)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                image_features = None
                logits_per_image = None
                logits_per_text = None
            else:
                raise ValueError("Either images or texts must be provided")
        
        # Compute loss if requested
        loss = None
        if return_loss and logits_per_image is not None:
            loss = self.clip_loss(logits_per_text)
        
        # Return outputs
        if return_dict:
            outputs = {
                'img_embeds': image_features,
                'text_embeds': text_features,
                'logits': logits_per_image,
                'logits_per_text': logits_per_text,
                'loss_value': loss
            }
            return outputs
        else:
            return image_features, text_features, logits_per_image
    
    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss."""
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0
    
    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for contrastive learning."""
        return nn.functional.cross_entropy(
            logits, torch.arange(len(logits), device=logits.device)
        )
    

class BioMedCLIPClassifier(nn.Module):
    """
    Zero-shot classifier using BioMedCLIP for medical image classification.
    """
    
    def __init__(
        self,
        biomedclip_model: BioMedCLIPModel,
        ensemble: bool = False,
        templates: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize BioMedCLIP classifier.
        
        Args:
            biomedclip_model: Pretrained BioMedCLIP model
            ensemble: Whether to use prompt ensembling
            templates: List of prompt templates (for compatibility with factory)
            **kwargs: Additional arguments (for compatibility)
        """
        super().__init__()
        self.model = biomedclip_model
        self.ensemble = ensemble
        self.templates = templates if templates else ["a medical image of {}"]
    
    def create_text_prompts(
        self,
        class_names: List[str]
    ) -> List[str]:
        """
        Create text prompts for classes using templates.
        
        Args:
            class_names: List of class names
            
        Returns:
            List of text prompts
        """
        prompts = []
        for class_name in class_names:
            for template in self.templates:
                prompts.append(template.format(class_name))
        
        return prompts
    
    def classify_with_templates(
        self,
        pixel_values: torch.Tensor,
        class_names: List[str],
        **kwargs
    ) -> Dict:
        """
        Classify images using templates to generate prompts.
        
        Args:
            pixel_values: Preprocessed image tensors
            class_names: List of class names
            
        Returns:
            Dictionary containing logits and class names
        """
        pixel_values = pixel_values.to(self.model.device)
        class_similarities = []
        
        for cls_name in class_names:
            # Generate prompts for this class using templates
            prompts = []
            for template in self.templates:
                prompts.append(template.format(cls_name))
            
            # Encode all prompts for this class
            template_similarities = []
            for prompt in prompts:
                # Encode text and image
                text_features = self.model.encode_text(prompt)
                image_features = self.model.encode_image(pixel_values)
                
                # Compute similarity
                similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
                template_similarities.append(similarity)
            
            # Aggregate similarities across templates
            template_similarities = torch.stack(template_similarities, dim=1)  # [batch, num_templates]
            
            if self.ensemble and len(self.templates) > 1:
                # Average across templates for ensembling
                cls_sim = torch.mean(template_similarities, dim=1)
            else:
                # Take max similarity across templates
                cls_sim = torch.max(template_similarities, dim=1)[0]
            
            class_similarities.append(cls_sim)
        
        # Stack similarities for all classes
        class_similarities = torch.stack(class_similarities, dim=1)  # [batch, num_classes]
        
        outputs = {
            'logits': class_similarities,
            'class_names': class_names,
        }
        
        return outputs
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        prompt_inputs: Dict[str, Dict],
        **kwargs
    ) -> Dict:
        """
        Forward pass for zero-shot classification.
        
        Args:
            pixel_values: Preprocessed image tensors
            prompt_inputs: Dictionary mapping class names to their text inputs
            
        Returns:
            Dictionary containing logits and class names
        """
        pixel_values = pixel_values.to(self.model.device)
        class_similarities = []
        class_names = []
        
        for cls_name, cls_text in prompt_inputs.items():
            inputs = {'pixel_values': pixel_values}
            
            # Handle tokenized inputs
            if 'input_ids' in cls_text:
                inputs['input_ids'] = cls_text['input_ids'].to(self.model.device)
            if 'attention_mask' in cls_text:
                inputs['attention_mask'] = cls_text['attention_mask'].to(self.model.device)
            
            # Get model outputs
            outputs = self.model(**inputs)
            logits = outputs['logits']
            
            # Aggregate similarities
            if self.ensemble:
                cls_sim = torch.mean(logits, 1)  # Prompt ensembling
            else:
                cls_sim = torch.max(logits, 1)[0]  # Take max similarity
            
            class_similarities.append(cls_sim)
            class_names.append(cls_name)
        
        class_similarities = torch.stack(class_similarities, 1)
        
        outputs = {
            'logits': class_similarities,
            'class_names': class_names,
        }
        
        return outputs


class BioMedCLIPFeatureExtractor(nn.Module):
    """
    Feature extractor using BioMedCLIP vision encoder.
    Can be used for downstream supervised learning tasks.
    """
    
    def __init__(
        self,
        biomedclip_model: BioMedCLIPModel,
        num_classes: int,
        freeze_encoder: bool = True,
        hidden_dim: Optional[int] = None
    ):
        """
        Initialize feature extractor with classification head.
        
        Args:
            biomedclip_model: Pretrained BioMedCLIP model
            num_classes: Number of output classes
            freeze_encoder: Whether to freeze the vision encoder weights
            hidden_dim: Optional hidden dimension for classification head
        """
        super().__init__()
        self.encoder = biomedclip_model
        self.num_classes = num_classes
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.model.parameters():
                param.requires_grad = False
        
        # Get feature dimension from the model
        with torch.no_grad():
            dummy_image = torch.randn(1, 3, 224, 224).to(self.encoder.device)
            features = self.encoder.encode_image(dummy_image, normalize=False)
            feature_dim = features.shape[-1]
        
        # Create classification head
        if hidden_dim:
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> Dict:
        """
        Forward pass for feature extraction and classification.
        
        Args:
            pixel_values: Preprocessed image tensors
            labels: Ground truth labels
            return_loss: Whether to compute and return loss
            
        Returns:
            Dictionary containing logits, embeddings, and optionally loss
        """
        outputs = defaultdict()
        
        # Extract features
        img_embeds = self.encoder.encode_image(pixel_values, normalize=False)
        
        # Classification
        logits = self.classifier(img_embeds)
        
        outputs['embedding'] = img_embeds
        outputs['logits'] = logits
        
        # Compute loss if labels provided
        if labels is not None and return_loss:
            labels = labels.to(self.encoder.device)
            if self.num_classes == 1:
                # Binary classification
                loss_fn = nn.BCEWithLogitsLoss()
                labels = labels.float().view(-1, 1)
            else:
                # Multi-class classification
                loss_fn = nn.CrossEntropyLoss()
                labels = labels.long()
            
            loss = loss_fn(logits, labels)
            outputs['loss_value'] = loss
        
        return outputs

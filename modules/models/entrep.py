import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, AutoTokenizer, AutoModelForMaskedLM
from timm.models.vision_transformer import VisionTransformer
from pathlib import Path
from functools import partial
from huggingface_hub import snapshot_download
from typing import Optional, Dict, Any, Union, List
from PIL import Image

from .base import TextEncoder, VisionEncoder
from .model import BaseVisionLanguageModel, BaseSupervisedClassifier, BaseZeroShotClassifier


class CLIPTextEncoder(TextEncoder):
    """CLIP Text Encoder"""
    
    def __init__(self, model_name: str = "medicalai/ClinicalBERT", 
                 feature_dim: int = 768, dropout_rate: float = 0.3, ckp_path: Optional[str] = None,
                 pretrained: bool = True):
        super().__init__(feature_dim)
        self.feature_dim = feature_dim
        
        if pretrained:
            try:
                self.text_model = AutoModelForMaskedLM.from_pretrained("local_model/clinical_bert", 
                                             use_safetensors=True, 
                                             local_files_only=True,
                                             trust_remote_code=True)
            except:
                # Fallback if safetensors fails
                self.text_model = AutoModelForMaskedLM.from_pretrained("local_model/clinical_bert", 
                                             use_safetensors=False, 
                                             local_files_only=True,
                                             trust_remote_code=True)
        else:
            # For testing - create model without pretrained weights
            self.text_model = AutoModelForMaskedLM(config)
            
        # Layer normalization and dropout
        self.ln = nn.LayerNorm(self.get_feature_dim())
        self.dropout = nn.Dropout(dropout_rate)
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.get_feature_dim(), self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )

        if ckp_path is not None:
            self.load_pretrained(ckp_path)

    def load_pretrained(self, model_path: str):
        """Load pretrained model from path"""
        self.text_model.load_state_dict(torch.load(model_path))

    def get_feature_dim(self) -> int:
        return self.text_model.config.hidden_size
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        text_outputs = self.text_model(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        output_hidden_states=True  # Get hidden states
    )
    
        # Get the last hidden layer
        hidden_states = text_outputs.hidden_states[-1]  # Shape: [batch, seq_len, hidden_dim]
        
        # Pooling: take [CLS] token (first token) or mean pooling
        embeddings = hidden_states[:, 0, :]
        embeddings = self.ln(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.projection(embeddings)
        text_features = embeddings / embeddings.norm(dim=-1, keepdim=True)
        if return_features:
            return embeddings
        return text_features

# Vision Encoder Implementations
class CLIPVisionEncoder(VisionEncoder):
    """CLIP Vision Encoder"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", 
                 feature_dim: int = 768, dropout_rate: float = 0.3, ckp_path: Optional[str] = None,
                 pretrained: bool = True):
        super().__init__(feature_dim)
        
        if pretrained:
            try:
                self.clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=False)
            except:
                # Fallback if safetensors fails
                self.clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
        else:
            # For testing - create model without pretrained weights
            from transformers import CLIPConfig
            config = CLIPConfig()
            self.clip_model = CLIPModel(config)
            
        self.vision_model = self.clip_model.vision_model
        
        # Layer normalization and dropout
        self.ln = nn.LayerNorm(self.get_feature_dim())
        self.dropout = nn.Dropout(dropout_rate)
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.get_feature_dim(), self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )

        if ckp_path is not None:
            self.load_pretrained(ckp_path)

    def get_feature_dim(self) -> int:
        return self.vision_model.config.hidden_size
    
    def forward(self, images: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        vision_outputs = self.vision_model(pixel_values=images)
        embeddings = vision_outputs.pooler_output
        embeddings = self.ln(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.projection(embeddings)
        if return_features:
            return embeddings
        return F.normalize(embeddings, dim=-1)

    
    def load_pretrained(self, model_path: str):
        """Load pretrained model from path"""
        self.clip_model.load_state_dict(torch.load(model_path))

class EndoViTVisionEncoder(VisionEncoder):
    """Complete EndoViT Vision Encoder with integrated backbone and head"""
    
    def __init__(self, 
                 model_name: str = "egeozsoy/EndoViT",
                 feature_dim: int = 768,
                 num_classes: int = 7,
                 dropout: float = 0.1,
                 freeze_backbone: bool = False,
                 ckp_path: Optional[str] = None):
        super().__init__(feature_dim)
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Initialize backbone with EndoViT
        self._init_backbone(model_name)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("🔒 Frozen backbone parameters")
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("🔓 Unfrozen backbone parameters")
        
        self.feature_projection = nn.Sequential(
            nn.Linear(self.backbone_feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        if ckp_path is not None:
            self.load_pretrained(ckp_path)

    def _init_backbone(self, model_name: str):
        """Initialize the backbone model"""
        model_path = snapshot_download(repo_id=model_name, revision="main")
        model_weights_path = Path(model_path) / "pytorch_model.bin"
        
        if model_weights_path.exists():
            self.backbone = VisionTransformer(
                patch_size=16, 
                embed_dim=self.feature_dim, 
                depth=12, 
                num_heads=12, 
                mlp_ratio=4, 
                qkv_bias=True, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            ).eval()
            
            model_weights = torch.load(model_weights_path, map_location='cpu', weights_only=False)
            if 'model' in model_weights:
                model_weights = model_weights['model']
                
            loading_info = self.backbone.load_state_dict(model_weights, strict=False)
            self.backbone_feature_dim = self.feature_dim
            print(f"✅ Successfully loaded pretrained EndoViT: {loading_info}")
        else:
            raise FileNotFoundError("EndoViT weights not found")
            
    def get_feature_dim(self) -> int:
        """Get feature dimension"""
        return self.feature_dim
    
    def _extract_backbone_features(self, x):
        """Extract features from backbone"""
        try:
            if hasattr(self.backbone, 'forward_features'):
                features = self.backbone.forward_features(x)
                if len(features.shape) == 3:  # [batch, seq_len, embed_dim]
                    return features[:, 0]  # CLS token
                else:
                    return features
            else:
                return self.backbone(x)
        except Exception as e:
            print(f"⚠️ Error in backbone forward pass: {e}")
            return self.backbone(x)
    
    def forward(self, images: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """Forward pass"""
        x = images
        backbone_features = self._extract_backbone_features(x)
        
        features = self.feature_projection(backbone_features)
        
        if return_features:
            return F.normalize(features, dim=-1)
        
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Get feature embeddings"""
        return self.forward(x, return_features=True)
    
    def get_backbone_features(self, x):
        """Get raw backbone features"""
        return self._extract_backbone_features(x)
    
    def load_pretrained(self, model_path: str):
        checkpoint = torch.load(model_path, map_location='cpu')
            
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict
        self.backbone.load_state_dict(state_dict, strict=False)
        print("✅ Checkpoint loaded successfully")

class DinoV2VisionEncoder(VisionEncoder):
    """Complete DinoV2 model"""
    def __init__(self, 
                 model_name: str = 'dinov2_vitb14',
                 feature_dim: int = 768,
                 num_classes: int = 7,
                 dropout: float = 0.1,
                 freeze_backbone: bool = False,
                 ckp_path: Optional[str] = None):
        super().__init__()
        
        # Initialize backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("🔒 Frozen backbone parameters")
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("🔓 Unfrozen backbone parameters")
        
        # Initialize feature projection layers (formerly DinoV2Head)
        self.feature_projection = nn.Sequential(
            nn.Linear(self.backbone.num_features, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Store config
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        if ckp_path is not None:
            self.load_pretrained(ckp_path)
        
    def forward(self, x, return_features=False):
        """Forward pass"""
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Project features
        features = self.feature_projection(backbone_features)
        
        if return_features:
            return features
        
        # Classify
        logits = self.classifier(features)
        return logits
    
    def get_feature_dim(self) -> int:
        return self.feature_dim
    
    def get_features(self, x):
        """Get feature embeddings"""
        return self.forward(x, return_features=True)
    
    def get_backbone_features(self, x):
        """Get raw backbone features"""
        return self.backbone(x)

    def load_pretrained(self, model_path: str):
        """Load pretrained model from path"""
        checkpoint = torch.load(model_path, map_location='cpu')
            
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict
        self.backbone.load_state_dict(state_dict, strict=False)
        print("✅ Checkpoint loaded successfully")

class ENTRepModel(nn.Module):
    """
    ENTRep base model that can combine different text and vision encoders
    Similar to MedCLIPModel structure
    """
    
    def __init__(
        self,
        text_encoder_type: str = 'clip',
        vision_encoder_type: str = 'endovit', 
        model_name: str = "openai/clip-vit-base-patch32",
        feature_dim: int = 768,
        dropout_rate: float = 0.3,
        dropout: float = 0.1,
        num_classes: int = 7,
        freeze_backbone: bool = False,
        text_checkpoint: Optional[str] = None,
        vision_checkpoint: Optional[str] = None,
        logit_scale_init_value: float = 0.07,
        pretrained: bool = True,
        **kwargs
    ):
        """
        Initialize ENTRep Model with flexible encoders
        
        Args:
            text_encoder_type: 'clip' or 'none'
            vision_encoder_type: 'clip', 'endovit', 'dinov2'
            model_name: Pretrained model name
            feature_dim: Feature dimension
            text_checkpoint: Checkpoint for text encoder
            vision_checkpoint: Checkpoint for vision encoder
            pretrained: Whether to load pretrained weights (for testing)
        """
        super().__init__()
        
        # Store encoder types
        self.text_encoder_type = text_encoder_type
        self.vision_encoder_type = vision_encoder_type
        self.feature_dim = feature_dim
        self.pretrained = pretrained
        
        # Initialize text encoder
        self.text_model = self._create_text_encoder(
            text_encoder_type, model_name, feature_dim, dropout_rate, text_checkpoint, pretrained
        )
        
        # Initialize vision encoder  
        self.vision_model = self._create_vision_encoder(
            vision_encoder_type, model_name, feature_dim, dropout_rate, dropout,
            num_classes, freeze_backbone, vision_checkpoint, pretrained
        )
        
        # Logit scale parameter for contrastive learning
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))
    
    def _create_text_encoder(
        self, 
        encoder_type: str, 
        model_name: str, 
        feature_dim: int,
        dropout_rate: float,
        checkpoint: Optional[str],
        pretrained: bool
    ) -> Optional[TextEncoder]:
        """Create text encoder based on type"""
        if encoder_type == 'clip':
            return CLIPTextEncoder(
                model_name=model_name,
                feature_dim=feature_dim,
                dropout_rate=dropout_rate,
                ckp_path=checkpoint,
                pretrained=pretrained
            )
        elif encoder_type is None or encoder_type == 'none':
            return None
        else:
            raise ValueError(f"Unsupported text encoder type: {encoder_type}")
    
    def _create_vision_encoder(
        self,
        encoder_type: str,
        model_name: str,
        feature_dim: int,
        dropout_rate: float,
        dropout: float,
        num_classes: int,
        freeze_backbone: bool,
        checkpoint: Optional[str],
        pretrained: bool
    ) -> VisionEncoder:
        """Create vision encoder based on type"""
        if encoder_type == 'clip':
            return CLIPVisionEncoder(
                model_name=model_name,
                feature_dim=feature_dim,
                dropout_rate=dropout_rate,
                ckp_path=checkpoint,
                pretrained=pretrained
            )
        # elif encoder_type == 'endovit':
        #     # EndoViT doesn't need pretrained CLIP models, so we can always create it
        #     return EndoViTVisionEncoder(
        #         model_name="egeozsoy/EndoViT",
        #         feature_dim=feature_dim,
        #         num_classes=num_classes,
        #         dropout=dropout,
        #         freeze_backbone=freeze_backbone,
        #         ckp_path=checkpoint
        #     )
        elif encoder_type == 'dinov2':
            # DinoV2 loads from torch.hub, separate from CLIP
            return DinoV2VisionEncoder(
                model_name='dinov2_vitb14',
                feature_dim=feature_dim,
                num_classes=num_classes,
                dropout=dropout,
                freeze_backbone=freeze_backbone,
                ckp_path=checkpoint
            )
        else:
            raise ValueError(f"Unsupported vision encoder type: {encoder_type}")
    
    def encode_text(self, input_ids=None, attention_mask=None):
        """Encode text inputs - similar to MedCLIP structure"""
        if self.text_model is None:
            raise NotImplementedError(f"Text encoding not supported with text_encoder_type='{self.text_encoder_type}'")
        
        # Move to GPU like MedCLIP
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        
        # Get text embeddings
        text_embeds = self.text_model(input_ids, attention_mask)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds
    
    def encode_image(self, pixel_values=None):
        """Encode image inputs - similar to MedCLIP structure"""
        # Handle different vision encoder types
        if self.vision_encoder_type == 'clip':
            # For CLIP vision encoder
            vision_output = self.vision_model(pixel_values, return_features=False)
        else:
            # For EndoViT and DinoV2 - get normalized features
            vision_output = self.vision_model.get_features(pixel_values)
        
        # Normalize embeddings
        img_embeds = vision_output / vision_output.norm(dim=-1, keepdim=True)
        return img_embeds
    
    def forward(self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        return_loss=None,
        **kwargs,
        ):
        """Forward pass - similar to MedCLIP structure"""
        # Move inputs to GPU
        if input_ids is not None:
            input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        if pixel_values is not None:
            pixel_values = pixel_values.cuda()

        # Get embeddings
        img_embeds = self.encode_image(pixel_values)
        if self.text_model is not None:
            text_embeds = self.encode_text(input_ids, attention_mask)
        else:
            text_embeds = None

        # Compute logits if both modalities present
        if text_embeds is not None:
            logits_per_image = self.compute_logits(img_embeds, text_embeds)
            logits_per_text = logits_per_image.t()
        else:
            logits_per_image = None
            logits_per_text = None

        # Compute loss
        if return_loss and logits_per_text is not None:
            loss = self.clip_loss(logits_per_text)
        else:
            loss = None

        return {'img_embeds': img_embeds, 'text_embeds': text_embeds,
                'logits': logits_per_image, 'loss_value': loss, 'logits_per_text': logits_per_text}

    def compute_logits(self, img_emb, text_emb):
        """Compute logits - similar to MedCLIP"""
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        """CLIP contrastive loss - similar to MedCLIP"""
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Contrastive loss function"""
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
    
    def get_encoder_info(self) -> Dict[str, str]:
        """Get information about encoders being used"""
        return {
            'text_encoder': self.text_encoder_type,
            'vision_encoder': self.vision_encoder_type,
            'vision_model_type': type(self.vision_model).__name__,
            'text_model_type': type(self.text_model).__name__ if self.text_model else 'None'
        }


class ENTRepClassifier(nn.Module):
    """
    ENTRep classifier for zero-shot classification - similar to PromptClassifier
    """
    def __init__(self, entrep_model, ensemble=False, **kwargs) -> None:
        super().__init__()
        self.model = entrep_model
        self.ensemble = ensemble

    def forward(self, pixel_values=None, prompt_inputs=None, **kwargs):
        """
        Forward pass for zero-shot classification
        
        Args:
            pixel_values: Image tensors
            prompt_inputs: Dict of {'class1':{'input_ids':...,'attention_mask':,...}, 'class2':...}
        """
        pixel_values = pixel_values.cuda()
        class_similarities = []
        class_names = []
        
        for cls_name, cls_text in prompt_inputs.items():
            inputs = {'pixel_values': pixel_values}
            for k in cls_text.keys(): 
                inputs[k] = cls_text[k].cuda()

            # Get ENTRep model outputs
            entrep_outputs = self.model(**inputs)
            logits = entrep_outputs['logits']

            # Calculate class similarity
            if self.ensemble:
                cls_sim = torch.mean(logits, 1)  # Prompt ensembling
            else:
                cls_sim = torch.max(logits, 1)[0]  # Max similarity
            
            class_similarities.append(cls_sim)
            class_names.append(cls_name)

        class_similarities = torch.stack(class_similarities, 1)
        outputs = {
            'logits': class_similarities,
            'class_names': class_names,
        }
        return outputs
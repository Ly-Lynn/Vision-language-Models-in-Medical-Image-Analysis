import torch
import torch.nn as nn
from typing import List
from .util import pil_to_tensor, tensor_to_pillow

class EvaluatePerturbation:
    def __init__(
        self,
        model: nn.Module,
        class_prompts: List[str], # (NUM_CLASSES x D)
        imgs: torch.Tensor=None,              # (B, 3, H, W)
        clean_pred_id: int=None               # index of clean prediction
    ):
        self.model = model
            
        self.class_text_feats = self.extract_centroid_vector(class_prompts)
        self.imgs = imgs
        self.clean_pred_id = clean_pred_id
        
    def set_data(self, image, clean_pred_id):
        self.img = image
        self.img_tensor = pil_to_tensor([image]).cuda()
        self.clean_pred_id = clean_pred_id
        
    
    @torch.no_grad() 
    def extract_centroid_vector(self, class_prompts): 
        class_features = [] 
        for class_name, item in class_prompts.items(): 
            text_feats = self.model.encode_text(item) 
            mean_feats = text_feats.mean(dim=0)
            class_features.append(mean_feats) 
            
        class_features = torch.stack(class_features) # NUM_ClASS x D 
        return class_features
            
        
    @torch.no_grad()
    def evaluate_blackbox(self, perturbations: torch.Tensor):
        adv_imgs = self.img_tensor + perturbations
        adv_imgs = torch.clamp(adv_imgs, 0, 1)
        adv_imgs = tensor_to_pillow(adv_imgs) # pillow image
        adv_feats = self.model.encode_image(adv_imgs)  # (B, D)
        sims = adv_feats @ self.class_text_feats.T     # (B, NUM_CLASSES)
        # Correct class similarity
        correct_sim = sims[:, self.clean_pred_id]

        # Max of other classes
        mask = torch.ones_like(sims, dtype=bool)
        mask[:, self.clean_pred_id] = False
        other_max_sim = sims[mask].view(sims.size(0), -1).max(dim=1, keepdim=True).values  # (B, 1)
        margin = correct_sim - other_max_sim
        return margin
    

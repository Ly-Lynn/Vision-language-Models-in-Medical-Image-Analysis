import torch
import torch.nn as nn
from typing import List


class EvaluatePerturbation:
    def __init__(
        self,
        model: nn.Module,
        class_texts: List[str], # (NUM_CLASSES x D)
        imgs: torch.Tensor,              # (B, 3, H, W)
        clean_pred_id: int               # index of clean prediction
    ):
        self.model = model
        with torch.no_grad(): 
            class_feats = self.model.encode_text(class_texts)
            
        self.class_text_feats = self.extract_centroid_vector(class_feats)
        self.imgs = imgs
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
        adv_imgs = self.imgs + perturbations
        adv_imgs = torch.clamp(adv_imgs, 0, 1)

        adv_feats = self.model.encode_image(adv_imgs)  # (B, D)
        sims = adv_feats @ self.class_text_feats.T     # (B, NUM_CLASSES)

        # Correct class similarity
        correct_sim = sims[:, self.clean_pred_id]

        # Max of other classes
        mask = torch.ones_like(sims, dtype=bool)
        mask[:, self.clean_pred_id] = False
        other_max_sim = sims[mask].view(sims.size(0), -1).max(dim=1).values

        margin = correct_sim - other_max_sim

        return margin

import torch
import torch.nn as nn
from pathlib import Path
from model_factory import DinoV2Model


model = DinoV2Model(
    model_name="dinov2_vitb14",
    feature_dim=768,
    num_classes=7,
    dropout=0.1,
    freeze_backbone=True
).cuda()

#load checkpoint
print("Loading model from checkpoint...")
checkpoint_path = Path("/datastore/elo/khoatn/Vision-language-Models-in-Medical-Image-Analysis/checkpoints/entrep_dino_b.pth")
checkpoint = torch.load(checkpoint_path)
missing_keys, unexpected_keys=model.load_state_dict(checkpoint["model_state_dict"])
print(f"Missing keys: {missing_keys}")
print(f"Unexpected keys: {unexpected_keys}")
model.eval()
print("Model loaded successfully.")

print("\n🧪 Testing model with dummy input...")
dummy_input = torch.randn(1, 3, 224, 224)
if torch.cuda.is_available():
    dummy_input = dummy_input.cuda()

# get feature
with torch.no_grad():
    output = model.get_features(dummy_input)
print("Output feature shape:", output.shape)
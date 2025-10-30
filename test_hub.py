import torch
#model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# print(model)

#import torch
from dinov2.models.vision_transformer import vit_large

# Tạo kiến trúc model (ViT-L/14)
model = vit_large(patch_size=14)

# Load checkpoint local
ckpt_path = "pretrained/dinov2/dinov2_vitl14_pretrain.pth"
state_dict = torch.load(ckpt_path, map_location="cpu")

# Một số checkpoint lưu trong key "model" hoặc "state_dict"
if "model" in state_dict:
        state_dict = state_dict["model"]

        model.load_state_dict(state_dict, strict=True)
        model.eval()


print(model)

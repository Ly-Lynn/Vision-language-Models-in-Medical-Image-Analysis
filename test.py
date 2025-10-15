from modules.dataset.factory import DatasetFactory
from modules.utils.constants import MODEL_TRANSFORMS, DEFAULT_TEMPLATES, RSNA_CLASS_PROMPTS, RSNA_CLASS_PROMPTS 
from modules.models.factory import ModelFactory, create_medclip, create_biomedclip, create_entrep
from modules.evaluator import ZeroShotEvaluator, TextToImageRetrievalEvaluator
from modules.utils.helpers import generate_rsna_class_prompts, generate_covid_class_prompts
from tqdm import tqdm
import numpy as np
import torch
import json

def _extract_label(dict_label):
    for i, (class_name, is_gt) in enumerate(dict_label.items()):
        if is_gt == 1:
            return i
        
n_prompt = 5
dataset_name = "rsna"
model_type = 'medclip'
transform = MODEL_TRANSFORMS[model_type]
batch_size = 128

DATA_ROOT = '/data2/elo/khoatn/Vision-language-Models-in-Medical-Image-Analysis/local_data'
dataset = DatasetFactory.create_dataset(
    dataset_name=dataset_name,
    model_type=model_type,
    data_root=DATA_ROOT,
    transform=None
)


model = ModelFactory.create_model(
    model_type=model_type,
    variant='base',
    pretrained=True 
)

class_prompts = generate_rsna_class_prompts(RSNA_CLASS_PROMPTS, n_prompt)
class_features = []
for class_name, item in class_prompts.items():
    text_feats = model.encode_text(item)
    mean_feats = text_feats.mean(dim=0)
    class_features.append(mean_feats) 
class_features = torch.stack(class_features) #  NUM_ClASS x D

all_preds = []
all_labels = []
with torch.no_grad():
    for start in tqdm(range(0, len(dataset), batch_size), desc="Infer"):
        end = min(start + batch_size, len(dataset))
        batch_imgs = []
        batch_lbl_indices = []

        for idx in range(start, end):
            img, label_dict = dataset[idx]
            img_tensor = transform(img)  
            batch_imgs.append(img_tensor)


            batch_lbl_indices.append(_extract_label(label_dict))



        batch_imgs = torch.stack(batch_imgs, dim=0).cuda()  # (B, C, H, W)
        batch_lbl_indices = torch.tensor(batch_lbl_indices, dtype=torch.long)

        img_feats = model.encode_image(batch_imgs)               # (B, D)
        sims = img_feats @ class_features.T                     # (B, NUM_CLASS)
        preds = sims.argmax(dim=-1).cpu()                       # (B,)

        all_preds.append(preds)
        all_labels.append(batch_lbl_indices)
        # break



all_preds = torch.cat(all_preds, dim=0)     # (N,)
all_labels = torch.cat(all_labels, dim=0)   # (N,)
acc = (all_preds == all_labels).float().mean().item()


results = [{"index": i, "class_pred_id": int(cls_id)} for i, cls_id in enumerate(all_preds.tolist())]
fname_results = f"evaluate_result/model_name={model_type}_dataset={dataset_name}_n_prompt={n_prompt}.json"
with open(fname_results, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Saved per-sample predictions to: {fname_results}")

# 2) Ghi lại dict class_prompts
fname_prompts = f"evaluate_result/model_name={model_type}_n_prompt={n_prompt}.json"
with open(fname_prompts, "w", encoding="utf-8") as f:
    json.dump(class_prompts, f, ensure_ascii=False, indent=2)
print(f"Saved class_prompts to: {fname_prompts}")


print(f"Total samples evaluated: {len(all_labels)}")
print(f"Accuracy: {acc:.4f}")
        






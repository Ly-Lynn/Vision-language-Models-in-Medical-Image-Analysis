from modules.dataset.factory import DatasetFactory
from modules.utils.constants import MODEL_TRANSFORMS, DEFAULT_TEMPLATES, RSNA_CLASS_PROMPTS, RSNA_CLASS_PROMPTS, ENTREP_CLASS_PROMPTS, ENTREP_TASKS 
from modules.models.factory import ModelFactory, create_medclip, create_biomedclip, create_entrep
from modules.evaluator import ZeroShotEvaluator, TextToImageRetrievalEvaluator
from modules.utils.helpers import generate_rsna_class_prompts, generate_covid_class_prompts
from tqdm import tqdm
import numpy as np
import torch
import yaml 
import json
import pandas as pd
from PIL import Image
from collections import OrderedDict
def _strip_prefix_from_state_dict(sd, prefixes=( 'module.', 'model.')):

    if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
        sd = sd['state_dict']

    new_sd = OrderedDict()
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        new_sd[nk] = v
    return new_sd


def get_entrep_data(path):
    df = pd.read_csv(path, sep=",")
    img_paths = df['image_path'].tolist()
    labels = {      
        'vocal-throat': df['vocal-throat'].tolist(), # [0, 1, ..]
        'nose': df['nose'].tolist(),
        'ear': df['ear'].tolist(),
        'throat': df['throat'].tolist(),
    }
    data = []
    for i in range(len(img_paths)):
        img_path = img_paths[i]
        img = Image.open(img_path)
        label_dict = {
            'vocal-throat': labels['vocal-throat'][i],
            'nose': labels['nose'][i],
            'ear': labels['ear'][i],
            'throat': labels['throat'][i],
        }
        data.append((img, label_dict))
        
    return data


# data = get_entrep_data("local_data/entrep/entrep_data.csv")
# print(data[0])
# raise
    
    
    
    
    


def _extract_label(dict_label):
    for i, (class_name, is_gt) in enumerate(dict_label.items()):
        if is_gt == 1:
            return i
        
n_prompt = 5
dataset_name = "rsna"
model_type = 'biomedclip'  # medclip, biomedclip, entrep
transform = MODEL_TRANSFORMS[model_type]
batch_size = 4

DATA_ROOT = 'local_data'
if dataset_name == "entrep":
    dataset = get_entrep_data("local_data/entrep/entrep_data.csv")
    
else:
    dataset = DatasetFactory.create_dataset(
        dataset_name=dataset_name,
        model_type=model_type,
        data_root=DATA_ROOT,
        transform=None
    )
# print(dataset[0])k
print(len(dataset))# raise

# ====================== Load model =====================
if model_type == "medclip":

    model = ModelFactory.create_model(
        model_type=model_type,
        variant='base',
        pretrained=True 
    )

elif model_type == "entrep":
    config_path = "configs/entrep_contrastive.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model_config = config.get('model', {})
        
    model = ModelFactory.create_model(
        model_type=model_type,
        variant='base',
        checkpoint="/datastore/elo/khoatn/Vision-language-Models-in-Medical-Image-Analysis/local_model/entrep_acmm/final thesis/vit_b_scratch/vitb_scratch.pt",
        # checkpoint=None,
        pretrained=False,
        **{k: v for k, v in model_config.items() if k != 'model_type' and k != "pretrained" and k != "checkpoint"}

        )
    
elif model_type == "biomedclip":
    model = ModelFactory.create_model(
        model_type=model_type,
        variant='base',
        pretrained=False,
        )
    checkpoint_path = "/datastore/elo/khoatn/Vision-language-Models-in-Medical-Image-Analysis/local_model/biomedclip/ssl_finetune.pt"
    checkpoint = torch.load(checkpoint_path)['model_state_dict']
    model.load_state_dict(checkpoint, strict=True)
    # raise
    
    
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# raise
# --------------------- Load class prompts ---------------------
if dataset_name == "rsna":
    # class_prompts = generate_rsna_class_prompts(RSNA_CLASS_PROMPTS, n_prompt)
    class_prompts = RSNA_CLASS_PROMPTS
elif dataset_name == "entrep":
    class_prompts = ENTREP_CLASS_PROMPTS
    
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


        # print(batch_imgs)
        # print(batch_lbl_indices)
        batch_imgs = torch.stack(batch_imgs, dim=0).cuda()  # (B, C, H, W)
        batch_lbl_indices = torch.tensor(batch_lbl_indices, dtype=torch.long)

        img_feats = model.encode_image(batch_imgs)               # (B, D)
        sims = img_feats @ class_features.T                     # (B, NUM_CLASS)
        preds = sims.argmax(dim=-1).cpu()                       # (B,)
        # print(preds)
        all_preds.append(preds)
        all_labels.append(batch_lbl_indices)
        # break



all_preds = torch.cat(all_preds, dim=0)     # (N,)
all_labels = torch.cat(all_labels, dim=0)   # (N,)
acc = (all_preds == all_labels).float().mean().item()


results = [
    {
        "index": i,
        'class_pred_id': int(cls_id),
        "gt_pred_id": int(gt_id)
    } for i, (cls_id, gt_id) in enumerate(zip(all_preds.tolist(), all_labels.tolist()))
]

fname_results = f"evaluate_result/model_name={model_type}_dataset={dataset_name}_n_prompt={n_prompt}.json"
with open(fname_results, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Saved per-sample predictions to: {fname_results}")

# 2) Ghi lại dict class_prompts
fname_prompts = f"evaluate_result/ssl_model_name={model_type}_dataset={dataset_name}_prompt.json"
with open(fname_prompts, "w", encoding="utf-8") as f:
    json.dump(class_prompts, f, ensure_ascii=False, indent=2)
print(f"Saved class_prompts to: {fname_prompts}")


print(f"Total samples evaluated: {len(all_labels)}")
print(f"Accuracy: {acc:.4f}")

class_names = list(class_prompts.keys())
print("Per-class accuracy:")
for i, class_name in enumerate(class_names):
    mask = (all_labels == i)
    if mask.sum() == 0:
        print(f"  {class_name:<25}: No samples")
        continue
    class_acc = (all_preds[mask] == all_labels[mask]).float().mean().item()
    print(f"  {class_name:<25}: {class_acc:.4f} ({mask.sum().item()} samples)")
    






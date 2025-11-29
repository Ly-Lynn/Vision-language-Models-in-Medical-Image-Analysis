from modules.dataset.factory import DatasetFactory
from modules.utils.constants import MODEL_TRANSFORMS, DEFAULT_TEMPLATES, RSNA_CLASS_PROMPTS, RSNA_CLASS_PROMPTS, SIZE_TRANSFORM, DATA_ROOT
from modules.models.factory import ModelFactory
from tqdm import tqdm
import numpy as np
import torch
import json
from modules.attack.attack import ES_1_Lambda, ES_1_Lambda_visual, RandomSearch, NESAttack
from modules.attack.evaluator import EvaluatePerturbation, DCTDecoder
from modules.attack.util import seed_everything 
import os
from torchvision import transforms
import yaml
import pandas as pd
from PIL import Image
from collections import OrderedDict
_toTensor = transforms.ToTensor()
def _extract_label(dict_label):
    for i, (class_name, is_gt) in enumerate(dict_label.items()):
        if is_gt == 1:
            return i

def _strip_prefix_from_state_dict(sd, prefixes=('visual.')):

    if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
        sd = sd['state_dict']

    new_sd = OrderedDict()
    for k, v in sd.items():
        nk = k
        # Bỏ lần lượt các prefix nếu có
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

def main(args):
    
    # ================================ Take dataset ================================ 
    dataset = DatasetFactory.create_dataset(
        dataset_name=args.dataset_name,
        model_type=args.model_name,
        data_root=DATA_ROOT,
        transform=None
    )

    
    size_transform = SIZE_TRANSFORM[args.model_name]




    # ================================ Load selected indices ================================
    with open(args.index_path, "r") as f:
        indxs = [int(line.strip()) for line in f.readlines()]

    if not args.end_idx:
        indxs = indxs[args.start_idx:]
    else:
        indxs = indxs[args.start_idx:args.end_idx]

    print("Len attack: ", len(indxs))

    
    
    # =========================================== Text prompt feature =========================================
    with open(args.prompt_path, "r") as f:
        class_prompts = json.load(f)
    
    
    # ========================================= MODEL =========================================
    if args.model_name == "medclip": # MEDCLIP
        model = ModelFactory.create_model(
            model_type=args.model_name,
            variant='base',
            pretrained=True
        )
        if args.visual_backbone_pretrained:
            checkpoint = torch.load(args.visual_backbone_pretrained)['model_state_dict'] 
            checkpoint = _strip_prefix_from_state_dict(checkpoint)
            not_matching_key = model.vision_model.load_state_dict(checkpoint)
            print("Incabable key: ", not_matching_key)
        
    elif args.model_name == "entrep": # ENTREP CLIP
        config_path = "configs/entrep_contrastive.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_config = config.get('model', {})
        
        checkpoint_path = args.pretrained
        model = ModelFactory.create_model(
            model_type=args.model_name,
            variant='base',
            # checkpoint=checkpoint_path,
            checkpoint=None,
            pretrained=False,
            **{k: v for k, v in model_config.items() if k != 'model_type' and k != "pretrained" and k != "checkpoint"}
            )

    
    elif args.model_name == "biomedclip":
        model = ModelFactory.create_model(
            model_type=args.model_name,
            variant='base',
            pretrained=False,
            )     

          
    if args.pretrained:
        checkpoint = torch.load(args.pretrained)['model_state_dict']
        # checkpoint = _strip_prefix_from_state_dict(checkpoint)
        not_matching_key = model.load_state_dict(checkpoint, strict=False)
        print("Incabable key: ", not_matching_key)
    model.eval()
    
    
    if args.dataset_name == "rsna":
        class_prompts = RSNA_CLASS_PROMPTS
    elif args.dataset_name == "entrep":
        class_prompts = ENTREP_CLASS_PROMPTS
        
    class_features = []
    for class_name, item in class_prompts.items():
        text_feats = model.encode_text(item)
        mean_feats = text_feats.mean(dim=0)
        class_features.append(mean_feats) 
    class_features = torch.stack(class_features) #  NUM_ClASS x D

    
    asr = 0
    l2 = 0
    # --------------------------- Main LOOP ------------------ 
    for index in tqdm(indxs):
        img_transfer_path = os.path.join(args.transfer_dir, str(index), "adv_img.png")
        adv_img = Image.open(img_transfer_path).convert("RGB")
        _, label_dict = dataset[index]
        label_id = _extract_label(label_dict)
        
        if args.mode == "post_transform": # knowing transform
            img_attack_tensor = _toTensor(adv_img).unsqueeze(0).cuda()
            img_feats = model.encode_posttransform_image(img_attack_tensor)
        
        elif args.mode == "pre_transform": # w/o knoiwng transform
            img_attack_tensor = _toTensor(adv_img).unsqueeze(0).cuda()
            img_feats = model.encode_pretransform_image(img_attack_tensor)
      
      
        # re-evaluation
        sims = img_feats @ class_features.T                     # (B, NUM_CLASS)
        adv_preds = sims.argmax(dim=-1).item()                    # (B,)
        if adv_preds != label_id:
            asr += 1
            
        
    print("Asr: ", asr / len(indxs))  
        
        
        
        
    

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Adversarial Attack Runner")

    # Dataset & model
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of dataset (e.g., rsna, chestxray, etc.)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model architecture (e.g., clip, biomedclip, etc.)")
    

    # transfer_path
    parser.add_argument("--transfer_dir", type=str)

    # Files
    parser.add_argument("--index_path", type=str, required=True,
                        help="Path to txt file containing selected indices (one per line)")
    parser.add_argument("--prompt_path", type=str, required=True,
                        help="Path to JSON file containing class prompts")
    

 
    parser.add_argument("--mode", type=str, default="pre_transform",
                        choices=["pre_transform", "post_transform"],
                        help="Attack mode: before or after model transform")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)


    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    
    
    # using decoder
    parser.add_argument("--f_ratio", type=float, default=None)
    parser.add_argument("--visual_backbone_mode", type=str, default='scratch', choices=['ssl', 'scratch'],)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--visual_backbone_pretrained', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    main(args)
    
    
# CUDA_VISIBLE_DEVICES=4 python main_atttack.py --dataset_name rsna --model_name medclip --index_path evaluate_result/selected_indices_covid_medclip.txt --prompt_path evaluate_result/model_name\=medclip_dataset\=rsna_prompt.json --attacker_name ES_1_1 --epsilon 0.03 --norm linf --mode pre_transform --seed 22520691

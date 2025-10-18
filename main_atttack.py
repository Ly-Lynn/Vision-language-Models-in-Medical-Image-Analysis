from modules.dataset.factory import DatasetFactory
from modules.utils.constants import MODEL_TRANSFORMS, DEFAULT_TEMPLATES, RSNA_CLASS_PROMPTS, RSNA_CLASS_PROMPTS, SIZE_TRANSFORM, DATA_ROOT
from modules.models.factory import ModelFactory
from tqdm import tqdm
import numpy as np
import torch
import json
from modules.attack.attack import ES_1_1, ES_1_Lambda, ES_Mu_Lambda, RandomSearchAttack
from modules.attack.evaluator import EvaluatePerturbation
from modules.attack.util import seed_everything 
import os

def _extract_label(dict_label):
    for i, (class_name, is_gt) in enumerate(dict_label.items()):
        if is_gt == 1:
            return i
        


def main(args):
    
    # -------------- Take dataset --------------------
    dataset = DatasetFactory.create_dataset(
        dataset_name=args.dataset_name,
        model_type=args.model_name,
        data_root=DATA_ROOT,
        transform=None
    )
    
    size_transform = SIZE_TRANSFORM[args.model_name]

    
    with open(args.index_path, "r") as f:
        indxs = [int(line.strip()) for line in f.readlines()]
    
    # ---------------- Text prompt feature -----------------
    with open(args.prompt_path, "r") as f:
        class_prompts = json.load(f)
    
    
    # ------------------ model -----------------------
    model = ModelFactory.create_model(
        model_type=args.model_name,
        variant='base',
        pretrained=True
    )
    
    
    # ----------------------- Evaluator ---------------
    evaluator = EvaluatePerturbation(
        model=model,
        class_prompts=class_prompts,
        mode=args.mode
    )
    
    # path dir save
    save_dir = os.path.join(args.out_dir, args.model_name, args.dataset_name, f"attack_name{args.attacker_name}_epsilon={args.epsilon}_mode={args.mode}_seed={args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    
    
    # ----------------- Attacker -------------------
    if args.attacker_name == "random_search":
        attacker = RandomSearchAttack(
            evaluator=evaluator,
            eps=args.epsilon,
            norm=args.norm,
            iterations=100,
            pop_size=20
        )
    
    elif args.attacker_name == "ES_1_1": # number of evaluation = iterations
        attacker = ES_1_1(
            evaluator=evaluator,
            eps=args.epsilon,
            norm=args.norm,
            iterations=2000,
        )
        
    elif args.attacker_name == "ES_1_Lambda": # number of evalation = ierations * lambda
        attacker = ES_1_Lambda(
            evaluator=evaluator,
            eps=args.epsilon,
            norm=args.norm,
            iterations=100,
            lam=20
        )
    elif args.attacker_name == "ES_Mu_Lambda": # number of evalation = ierations * lambda
        attacker = ES_Mu_Lambda(
            evaluator=evaluator,
            eps=args.epsilon,
            norm=args.norm,
            iterations=100,
            lam=15,
            mu=5
        )
                
    # --------------------------- Main LOOP ------------------ 
    for index in tqdm(indxs):
        img, label_dict = dataset[index]
        label_id = _extract_label(label_dict)

        if args.mode == "post_transform": # knowing transform
            img_attack = size_transform(img).convert("RGB")
        elif args.mode == "pre_transform": # w/o knoiwng transform
            img_attack = img.convert("RGB")
        # print("Size spacce attack: ", img_attack.size)
        
        
        
        # re-evaluation
        img_feats = model.encode_image([img])
        sims = img_feats @ evaluator.class_text_feats.T                     # (B, NUM_CLASS)
        clean_preds = sims.argmax(dim=-1).item()                    # (B,)
        # assert preds == label_id
        # print("Clean preds: ", preds)

        # main attack
        attacker.evaluator.set_data(
            image=img_attack,
            clean_pred_id=clean_preds
        )
        
        result = attacker.run()
        delta = result['best_delta']
        adv_imgs, pil_adv_imgs = evaluator.take_adv_img(delta)
        # pil_adv_imgs[0].save("test.png")
        
        # recheck
        if args.mode == "post_transform": # knowing transform
            img_feats = model.encode_posttransform_image(adv_imgs) # (B, NUM_CLASS)
        elif args.mode == "pre_transform":
            img_feats = model.encode_pretransform_image(adv_imgs)  # (B, D)
        
        sims = img_feats @ evaluator.class_text_feats.T                     # (B, NUM_CLASS)
        adv_preds = sims.argmax(dim=-1).item()                    # (B,)
        # print("Adv preds: ", preds)
        
        # save_dir
        index_dir = os.path.join(save_dir, str(index))
        os.makedirs(index_dir, exist_ok=True)
        pil_adv_imgs[0].save(os.path.join(index_dir, f'adv_img.png'))
        img_attack.save(os.path.join(index_dir, "clean_img.png"))
        
        info = {
            'clean_pred': clean_preds,
            'adv_pred': adv_preds,
            'gt': label_id,
            'success_iterations': len(result['history'])
        }
        with open(os.path.join(index_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=4)
            
            
        
        
        
        
        
        
    

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Adversarial Attack Runner")

    # Dataset & model
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of dataset (e.g., rsna, chestxray, etc.)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model architecture (e.g., clip, biomedclip, etc.)")
    
    # Files
    parser.add_argument("--index_path", type=str, required=True,
                        help="Path to txt file containing selected indices (one per line)")
    parser.add_argument("--prompt_path", type=str, required=True,
                        help="Path to JSON file containing class prompts")
    
    # Attack configuration
    parser.add_argument("--attacker_name", type=str, required=True,
                        choices=["random_search", "ES_1_1", "ES_1_Lambda", "ES_Mu_Lambda", "PGD"],
                        help="Name of attacker algorithm")
    parser.add_argument("--epsilon", type=float, default=8/255,
                        help="Maximum perturbation magnitude (default: 8/255)")
    parser.add_argument("--norm", type=str, default="linf",
                        choices=["linf", "l2"],
                        help="Norm constraint type")
    parser.add_argument("--mode", type=str, default="pre_transform",
                        choices=["pre_transform", "post_transform"],
                        help="Attack mode: before or after model transform")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    
    # outdir
    parser.add_argument("--out_dir", type=str, default="attack_dir")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    main(args)
    
    
# CUDA_VISIBLE_DEVICES=4 python main_atttack.py --dataset_name rsna --model_name medclip --index_path evaluate_result/selected_indices_covid_medclip.txt --prompt_path evaluate_result/model_name\=medclip_dataset\=rsna_prompt.json --attacker_name ES_1_1 --epsilon 0.03 --norm linf --mode pre_transform --seed 22520691
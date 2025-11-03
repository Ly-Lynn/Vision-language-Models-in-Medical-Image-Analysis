from modules.dataset.factory import DatasetFactory
from modules.utils.constants import MODEL_TRANSFORMS, DEFAULT_TEMPLATES, RSNA_CLASS_PROMPTS, RSNA_CLASS_PROMPTS, SIZE_TRANSFORM, DATA_ROOT
from modules.models.factory import ModelFactory
from tqdm import tqdm
import numpy as np
import torch
import json
from modules.attack.attack import ES_1_Lambda, ES_1_Lambda_visual
from modules.attack.evaluator import EvaluatePerturbation, DCTDecoder
from modules.attack.util import seed_everything 
import os
from torchvision import transforms
import yaml

_toTensor = transforms.ToTensor()
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
    
    # data[i]
    # img, label_dict = dataset[index]
    
    
    
    size_transform = SIZE_TRANSFORM[args.model_name]

    
    with open(args.index_path, "r") as f:
        indxs = [int(line.strip()) for line in f.readlines()]

    if not args.end_idx:
        indxs = indxs[args.start_idx:]
    else:
        indxs = indxs[args.start_idx:args.end_idx]
    # take 100 lớp đầu, take 100 lớp sau:
    # indxs_0 = indxs[:100]
    # indxs_1 = indxs[500:600]
    # indxs = indxs_0 + indxs_1
    print("Len attack: ", len(indxs))
    
    # ---------------- Text prompt feature -----------------
    with open(args.prompt_path, "r") as f:
        class_prompts = json.load(f)
    
    
    # ------------------ model -----------------------
    if args.model_name == "medclip":
        model = ModelFactory.create_model(
            model_type=args.model_name,
            variant='base',
            pretrained=True
        )
        
    elif args.model_name == "entrep":
        config_path = "configs/entrep_contrastive.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_config = config.get('model', {})
            
        model = ModelFactory.create_model(
            model_type=args.model_name,
            variant='base',
            # checkpoint="checkpoints/entrep_checkpoint.pt",
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
         
    model.eval()
    
    # -------------- Decoder ------------
    decoder = None
    if args.f_ratio:
        decoder = DCTDecoder(
            f_ratio=args.f_ratio
        )
    
    
    # ----------------------- Evaluator ---------------
    evaluator = EvaluatePerturbation(
        model=model,
        class_prompts=class_prompts,
        mode=args.mode,
        decoder=decoder,
        eps=args.epsilon,
        norm=args.norm
    )
    
    # path dir save
    save_dir = os.path.join(args.out_dir, args.model_name, args.dataset_name, f"dct_f={args.f_ratio}_attack_name{args.attacker_name}_epsilon={args.epsilon}_norm={args.norm}_mode={args.mode}_seed={args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    
    
    # ----------------- Attacker -------------------
    if args.attacker_name == "ES_1_Lambda": # number of evalation = ierations * lambda
        attacker = ES_1_Lambda(
            evaluator=evaluator,
            eps=args.epsilon,
            norm=args.norm,
            max_evaluation=args.max_evaluation,
            lam=args.lamda
        )
        
    elif args.attacker_name == "ES_1_Lambda_visual": # number of evalation = ierations * lambda
        attacker = ES_1_Lambda_visual(
            evaluator=evaluator,
            eps=args.epsilon,
            norm=args.norm,
            max_evaluation=args.max_evaluation,
            lam=args.lamda,
            _bs_steps=args.bs_steps, 
            additional_eval=args.additional_eval
        )

                
    # --------------------------- Main LOOP ------------------ 
    for index in tqdm(indxs):
        img, label_dict = dataset[index]
        label_id = _extract_label(label_dict)

        if args.mode == "post_transform": # knowing transform
            img_attack = size_transform(img).convert("RGB")
            img_attack_tensor = _toTensor(img_attack).unsqueeze(0).cuda()
            img_feats = model.encode_posttransform_image(img_attack_tensor)
        
        elif args.mode == "pre_transform": # w/o knoiwng transform
            img_attack = img.convert("RGB")
            img_attack_tensor = _toTensor(img_attack).unsqueeze(0).cuda()
            img_feats = model.encode_pretransform_image(img_attack_tensor)
      
      
        # re-evaluation
        sims = img_feats @ evaluator.class_text_feats.T                     # (B, NUM_CLASS)
        clean_preds = sims.argmax(dim=-1).item()                    # (B,)
        print("Clean preds: ", clean_preds)
        print("Label_id: ", label_id)
        # raise

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
                        choices=["random_search", "ES_1_1", "ES_1_Lambda", "ES_1_Lambda_visual", "ES_Mu_Lambda", "PGD"],
                        help="Name of attacker algorithm")
    parser.add_argument("--epsilon", type=float, default=8/255,
                        help="Maximum perturbation magnitude (default: 8/255)")
    parser.add_argument("--norm", type=str, default="linf",
                        choices=["linf", "l2"],
                        help="Norm constraint type")
    parser.add_argument("--max_evaluation", type=int, default=10000)
    parser.add_argument("--lamda", type=int, default=50)
    parser.add_argument("--bs_steps", type=int, default=20)
    parser.add_argument("--additional_eval", type=int, default=200)
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
    
    # outdir
    parser.add_argument("--out_dir", type=str, default="attack_new")

    # using decoder
    parser.add_argument("--f_ratio", type=float, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    main(args)
    
    
# CUDA_VISIBLE_DEVICES=4 python main_atttack.py --dataset_name rsna --model_name medclip --index_path evaluate_result/selected_indices_covid_medclip.txt --prompt_path evaluate_result/model_name\=medclip_dataset\=rsna_prompt.json --attacker_name ES_1_1 --epsilon 0.03 --norm linf --mode pre_transform --seed 22520691

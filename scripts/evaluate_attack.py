import os
import json
from tqdm import tqdm
import json
import os
from PIL import Image
import torchvision.transforms.functional as F
import torch
from tqdm import tqdm
# config
seed = 22520691
model_name = "entrep"
epsilons = [
    0.03, 
    # 0.05, 
    # 0.08,
    # 0.1
    ]
lamb = 50
settings = [
    # "post_transform", 
    "pre_transform"
            ]
dataset_name = "entrep"
visual_backbone_mode = 'scratch'
sample_id_path = "/datastore/elo/khoatn/Vision-language-Models-in-Medical-Image-Analysis/scripts/common_entrep.txt"
with open(sample_id_path, "r") as f:
    sample_ids = [line.strip() for line in f if line.strip()]
# sample_ids =  os.listdir(dir)   
for epsilon in epsilons:
    for setting in settings:
        dir = "/datastore/elo/khoatn/Vision-language-Models-in-Medical-Image-Analysis/attack_tanh_transform/{}/{}/{}_dct_f=None_attack_nameES_1_Lambda_epsilon={}_norm=linf_mode={}_seed={}".format(
        # dir = "/datastore/elo/khoatn/Vision-language-Models-in-Medical-Image-Analysis/attack_tanh_transform/{}/{}/dct_f=None_attack_nameES_1_Lambda_epsilon={}_norm=linf_mode={}_seed={}".format(
            model_name, dataset_name, visual_backbone_mode, epsilon, setting, seed
            # model_name, dataset_name, epsilon, setting, seed
        )
        if not os.path.exists(dir):
            print(f"Not exists: {epsilon} - {setting}")
            continue

        len_ = len(sample_ids)
        success = 0
        success_iter = 0
        l2 = 0
        # for sample_id in tqdm(os.listdir(dir)):
        for sample_id in tqdm(sample_ids):
            info_path = os.path.join(dir, sample_id, 'info.json')
            clean_path = os.path.join(dir, sample_id, 'clean_img.png')
            adv_path = os.path.join(dir, sample_id, 'adv_img.png')
            
            with open(info_path, "r") as f:
                data = json.load(f)
            
            # if data['clean_pred'] != data['adv_pred'] or data['success_iterations'] < 201:        
            # if data['clean_pred'] != data['adv_pred']:        
            if data['success_iterations'] < 201:
                success += 1

                # print(sample_id, data['clean_pred'], data['adv_pred'], data['success_iterations'])
                
                # continue
                
            clean_img = Image.open(clean_path).convert('RGB')
            adv_img = Image.open(adv_path).convert('RGB')
            clean_tensor = F.to_tensor(clean_img)
            adv_tensor = F.to_tensor(adv_img)
            
            l2_norm = torch.norm((adv_tensor - clean_tensor).view(-1), p=2)
            success_iter += data['success_iterations'] * lamb + 1
            l2 += l2_norm

        data = {
            'asr': success / len_,
            'l2': float(l2.item()) / len_,
            'success_evaluate': success_iter / len_
        }
        print(f"{epsilon} - {setting}",data)
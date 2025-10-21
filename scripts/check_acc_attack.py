import json
import os
from PIL import Image
dir = r"/datastore/elo/khoatn/Vision-language-Models-in-Medical-Image-Analysis/attack_dir/medclip/rsna/dct_f=0.03125_attack_nameES_1_Lambda_epsilon=0.03_mode=pre_transform_seed=22520692"
test_dir = r"/datastore/elo/khoatn/Vision-language-Models-in-Medical-Image-Analysis/attack_dir/medclip/rsna/attack_nameES_1_Lambda_epsilon=0.03_mode=post_transform_seed=22520691"
len = len(os.listdir(dir))
import torchvision.transforms.functional as F
import torch
from tqdm import tqdm

# Chuyển sang tensor [0,1]


# Đảm bảo cùng shape

print(len)

success = 0
success_iter = 0
l2 = 0
for sample_id in tqdm(os.listdir(dir)):
    info_path = os.path.join(dir, sample_id, 'info.json')
    clean_path = os.path.join(dir, sample_id, 'clean_img.png')
    adv_path = os.path.join(dir, sample_id, 'adv_img.png')
    
    with open(info_path, "r") as f:
        data = json.load(f)
    if data['clean_pred'] != data['adv_pred']:
        success += 1
        
    clean_img = Image.open(clean_path).convert('RGB')
    adv_img = Image.open(adv_path).convert('RGB')
    clean_tensor = F.to_tensor(clean_img)
    adv_tensor = F.to_tensor(adv_img)
    
    l2_norm = torch.norm((adv_tensor - clean_tensor).view(-1), p=2)

    

    success_iter += data['success_iterations'] + 1
    l2 += l2_norm
    
print('success rate: ', success / len)
print('success_iter rate: ', success_iter / len)
print('l2 norm rate: ', l2 / len)

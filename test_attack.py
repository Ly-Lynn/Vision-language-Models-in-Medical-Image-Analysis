from modules.dataset.factory import DatasetFactory
from modules.utils.constants import MODEL_TRANSFORMS, DEFAULT_TEMPLATES, RSNA_CLASS_PROMPTS, RSNA_CLASS_PROMPTS
from modules.models.factory import ModelFactory, create_medclip, create_biomedclip, create_entrep
from modules.evaluator import ZeroShotEvaluator, TextToImageRetrievalEvaluator
from modules.utils.helpers import generate_rsna_class_prompts, generate_covid_class_prompts
from tqdm import tqdm
import numpy as np
import torch
import json
from modules.attack.attack import RandomSearchAttack
from modules.attack.evaluator import EvaluatePerturbation
from modules.attack.util import seed_everything 

def _extract_label(dict_label):
    for i, (class_name, is_gt) in enumerate(dict_label.items()):
        if is_gt == 1:
            return i

seed_everything(22520591)
n_prompt = 5
dataset_name = "rsna"
model_type = 'medclip'
epsilon = 0.1
iterations = 200
pop_size = 50
norm = 'linf'
DATA_ROOT = '/datastore/elo/khoatn/Vision-language-Models-in-Medical-Image-Analysis/local_data'
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

# evalu
evaluator = EvaluatePerturbation(
    model=model,
    class_prompts=class_prompts,
)
attacker = RandomSearchAttack(
    evaluator=evaluator,
    eps=epsilon,
    norm=norm,
    iterations=iterations,
    pop_size=pop_size
)
# attacker = ESAttack(
#     evaluator=evaluator,
#     eps=epsilon,
#     norm=norm,
#     iterations=iterations,
#     pop_size=pop_size
# )



with torch.no_grad():
    
    for i in range(len(dataset)):
        img, label_dict = dataset[i]
        img = img.convert("RGB")
        label_id = _extract_label(label_dict)
        img_feats = model.encode_image([img])
        sims = img_feats @ evaluator.class_text_feats.T                     # (B, NUM_CLASS)
        preds = sims.argmax(dim=-1)                    # (B,)
        print("Label_id: ", label_id)
        print("preds: ", preds)
        attacker.evaluator.set_data(
            image=img, 
            clean_pred_id=preds
        )
        
        result = attacker.run()
        delta = result['best_delta']
        adv_imgs, pil_adv_imgs = evaluator.take_adv_img(delta)
        img_feats = model.encode_image(pil_adv_imgs) # (B, NUM_CLASS)
        sims = img_feats @ evaluator.class_text_feats.T                     # (B, NUM_CLASS)
        preds = sims.argmax(dim=-1)                    # (B,)
        print("Adv preds: ", preds)
        pil_adv_imgs[0].save('test_adv.png')
        img.save('test.png')
        
        break
        # Attack module
        





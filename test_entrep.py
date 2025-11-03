from modules.models.factory import ModelFactory
from modules.dataset.factory import DatasetFactory
from modules.utils.constants import MODEL_TRANSFORMS, DEFAULT_TEMPLATES, RSNA_CLASS_PROMPTS, RSNA_CLASS_PROMPTS, SIZE_TRANSFORM, DATA_ROOT
import yaml

config_path = "configs/entrep_contrastive.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
    
model_config = config.get('model', {})

model_name = "entrep"
dataset_name = "entrep"
model = ModelFactory.create_model(
    model_type=model_name,
    variant='base',
    checkpoint="checkpoints/entrep_checkpoint.pt",
    pretrained=False,
    **{k: v for k, v in model_config.items() if k != 'model_type' and k != "pretrained" and k != "checkpoint"}

)

model.eval()

dataset_name = "entrep"
dataset = DatasetFactory.create_dataset(
    dataset_name=dataset_name,
    model_type=model_name,
    data_root=DATA_ROOT,
    transform=None

)

img, label = dataset[100]
# img.save("test.png")
print(label)

img_fea = model.encode_image([img])
print(img_fea.shape)
text_fea = model.encode_text(["nose", "ear", "throat", "voice-throat"])
print(text_fea.shape)
# print(img_fea @ text_fea.T)


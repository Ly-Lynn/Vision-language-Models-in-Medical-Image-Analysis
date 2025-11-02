from modules.models.factory import ModelFactory
from modules.dataset.factory import DatasetFactory
from modules.utils.constants import MODEL_TRANSFORMS, DEFAULT_TEMPLATES, RSNA_CLASS_PROMPTS, RSNA_CLASS_PROMPTS, SIZE_TRANSFORM, DATA_ROOT


model_name = "entrep"
dataset_name = "entrep"
model = ModelFactory.create_model(
    model_type=model_name,
    variant='base',
    pretrained=True
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
img.save("test.png")
print(label)

img_fea = model.encode_image([img])
text_fea = model.encode_text(["nose", "ear", "throat", "voice-throat"])

print(img_fea @ text_fea.T)


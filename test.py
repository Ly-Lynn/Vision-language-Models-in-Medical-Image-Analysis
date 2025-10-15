# dataset
from modules.dataset.factory import DatasetFactory
from modules.utils.constants import MODEL_TRANSFORMS

dataset_name = "covid"
model_type = 'medclip'
transform = MODEL_TRANSFORMS[model_type]

DATA_ROOT = '/data2/elo/khoatn/Vision-language-Models-in-Medical-Image-Analysis/local_data'
dataset = DatasetFactory.create_dataset(
    dataset_name=dataset_name,
    model_type=model_type,
    data_root=DATA_ROOT,
    transform=None
)

img, label = dataset[0]
img_tensor = transform(img)
print("Image tensor shape: ", img_tensor.shape)
print("Image shape: ", img.size)
img.save("testing_covid.png")
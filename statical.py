from modules.dataset.factory import DatasetFactory
from modules.utils.constants import MODEL_TRANSFORMS
from tqdm import tqdm
import numpy as np

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

print(f"Dataset: {dataset_name}")
print(f"Total samples: {len(dataset)}")

labels = [0, 0]
widths, heights = [], []
for i in tqdm(range(len(dataset)), desc="Scanning dataset"):
    img, label = dataset[i]
    img.save(f"{str(label)}.png")
    break

    for i, (key, item) in enumerate(label.items()):
        if item == 1:
            gt_id = i
    labels[gt_id] += 1
    
    
    w, h = img.size
    widths.append(w)
    heights.append(h)


print(labels)

widths = np.array(widths)
heights = np.array(heights)

print("\n=== Dataset Statistics ===")
print(f"Number of images: {len(dataset)}")
print(f"Width  - min: {widths.min()}, max: {widths.max()}, mean: {widths.mean():.2f}")
print(f"Height - min: {heights.min()}, max: {heights.max()}, mean: {heights.mean():.2f}")
print(f"Most common size: ({int(np.median(widths))}, {int(np.median(heights))})")

img, label = dataset[0]
img_tensor = transform(img)
print("\nExample image tensor shape:", img_tensor.shape)
print("Original image size:", img.size)
img.save("testing_covid.png")

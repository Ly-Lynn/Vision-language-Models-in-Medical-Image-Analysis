import argparse
import os
from collections import OrderedDict

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import open_clip
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from modules.dataset.factory import DatasetFactory
from modules.utils.constants import RSNA_CLASS_PROMPTS  # nếu muốn dùng sẵn, còn không thì ta override

# =========================================================
# Utils
# =========================================================

def _strip_prefix_from_state_dict(sd, prefixes=("module.", "model.")):
    """
    Remove common wrappers (e.g., 'module.') from checkpoints trained with DataParallel/Lightning.
    """
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    new_sd = OrderedDict()
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        new_sd[nk] = v
    return new_sd


def _extract_label(label_dict):
    """
    Input: dict_label = {class_name: 0/1, ...}
    Return: index của class được gán là 1 theo thứ tự duyệt dict.
    Giả định mỗi sample chỉ có đúng 1 class = 1.
    """
    for i, (_, is_gt) in enumerate(label_dict.items()):
        if is_gt == 1:
            return i
    return -1  # để dễ debug nếu label lỗi


def get_rsna_dataset(data_root, model_type="biomedclip"):
    """
    Dataset RSNA cố định, dùng DatasetFactory của bạn.
    Giả định DatasetFactory.create_dataset trả về (PIL.Image, label_dict).
    """
    dataset = DatasetFactory.create_dataset(
        dataset_name="rsna",
        model_type=model_type,
        data_root=data_root,
        transform=None,  # dùng preprocess của open_clip phía dưới
    )
    return dataset


def get_class_prompts_rsna():
    """
    Prompts cho pneumonia vs normal.
    Có thể thay bằng RSNA_CLASS_PROMPTS nếu bạn muốn dùng template phức tạp hơn.
    """
    class_prompts = {
        "Pneumonia": [
            "Pneumonia lung X-ray",
            "Chest X-ray showing pneumonia",
            "Lung consolidation consistent with pneumonia",
            "Chest radiograph with pneumonia findings",
            "Abnormal lung X-ray with pneumonia"
        ],
        "Normal": [
            "Normal lung X-ray",
            "Clear chest X-ray with no abnormalities",
            "Normal chest radiograph",
            "Healthy lungs on X-ray",
            "No signs of pneumonia on chest X-ray"
        ],
    }
    return class_prompts


def encode_class_prompts(model, tokenizer, class_prompts, device):
    """
    Encode toàn bộ text prompts cho từng class.

    class_prompts: dict[class_name] = list[text_prompt]

    Trả về:
        class_features: (C, D)  - mean feature per class (prototype)
        all_text_feats: (T, D)  - feature của từng prompt
        all_text_labels: (T,)   - class id cho từng prompt
    """
    class_features = []
    all_text_feats = []
    all_text_labels = []

    class_names = list(class_prompts.keys())

    for cls_idx, class_name in enumerate(class_names):
        prompts = class_prompts[class_name]
        text_tokens = tokenizer(prompts).to(device)

        with torch.no_grad():
            text_feats = model.encode_text(text_tokens)  # (n_prompt, D)

        mean_feats = text_feats.mean(dim=0)             # (D,)
        class_features.append(mean_feats)

        all_text_feats.append(text_feats.cpu())
        all_text_labels.extend([cls_idx] * text_feats.shape[0])

    class_features = torch.stack(class_features, dim=0)                # (C, D)
    all_text_feats = torch.cat(all_text_feats, dim=0)                  # (T, D)
    all_text_labels = torch.tensor(all_text_labels, dtype=torch.long)  # (T,)

    return class_features, all_text_feats, all_text_labels


def load_open_clip_model(args, device):
    """
    Load open_clip + optional custom checkpoint.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained="openai",
    )
    model.to(device)

    if args.pretrained is not None and len(args.pretrained) > 0:
        print(f"Loading custom checkpoint from: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu")
        state_dict = _strip_prefix_from_state_dict(ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model_name)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    return model, preprocess, tokenizer


def plot_pca_embeddings(
    img_feats,
    img_labels,
    class_features,
    class_names,
    text_feats,
    text_labels,
    save_path=None,
    max_points=5000,
):
    """
    Vẽ PCA 2D cho:
        - image embeddings (chấm tròn, màu theo class)
        - từng text prompt (tam giác, màu theo class)
        - class prototype (X đen, có nhãn)

    img_feats:      (N, D) tensor (CPU)
    img_labels:     (N,)   tensor (CPU, long)
    class_features: (C, D) tensor (CPU)
    class_names:    list[str], len = C
    text_feats:     (T, D) tensor (CPU)
    text_labels:    (T,)   tensor (CPU, long)
    """
    img_feats_np = img_feats.numpy()
    img_labels_np = img_labels.numpy()
    class_feats_np = class_features.numpy()
    text_feats_np = text_feats.numpy()
    text_labels_np = text_labels.numpy()

    # Optional: sample bớt ảnh cho plot đỡ dày
    n_img = img_feats_np.shape[0]
    if n_img > max_points:
        idx = np.random.choice(n_img, size=max_points, replace=False)
        img_feats_np = img_feats_np[idx]
        img_labels_np = img_labels_np[idx]
        n_img = img_feats_np.shape[0]
        print(f"Subsampled images to {n_img} points for PCA plotting.")

    n_class = class_feats_np.shape[0]
    n_text = text_feats_np.shape[0]

    combined = np.concatenate(
        [img_feats_np, class_feats_np, text_feats_np],
        axis=0,
    )

    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)

    img_2d = combined_2d[:n_img]
    class_2d = combined_2d[n_img:n_img + n_class]
    text_2d = combined_2d[n_img + n_class:]

    plt.figure(figsize=(8, 6))

    # ===== Images: chấm nhỏ, màu theo class =====
    num_classes = len(class_names)
    for cls_idx in range(num_classes):
        mask = (img_labels_np == cls_idx)
        if mask.sum() == 0:
            continue
        plt.scatter(
            img_2d[mask, 0],
            img_2d[mask, 1],
            s=8,
            alpha=0.4,
            label=f"{class_names[cls_idx]} (images)",
        )

    # ===== Text prompts: tam giác, màu theo class =====
    for cls_idx in range(num_classes):
        mask = (text_labels_np == cls_idx)
        if mask.sum() == 0:
            continue
        plt.scatter(
            text_2d[mask, 0],
            text_2d[mask, 1],
            s=40,
            marker="^",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
            label=f"{class_names[cls_idx]} (text prompts)",
        )

    # ===== Class prototypes: X đen, có nhãn =====
    plt.scatter(
        class_2d[:, 0],
        class_2d[:, 1],
        s=120,
        marker="X",
        edgecolor="black",
        linewidth=1.5,
        c="black",
        label="Class prototypes (mean text)",
    )

    for i, name in enumerate(class_names):
        plt.text(
            class_2d[i, 0],
            class_2d[i, 1],
            f" {name}",
            fontsize=10,
            weight="bold",
        )

    plt.xlabel("PCA dim 1")
    plt.ylabel("PCA dim 2")
    plt.title("PCA of image and text embeddings on RSNA (Pneumonia vs Normal)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Saved PCA plot to: {save_path}")
    else:
        plt.show()


# =========================================================
# Main logic: chỉ load RSNA, embed, rồi vẽ PCA
# =========================================================

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Dataset (RSNA cố định)
    dataset = get_rsna_dataset(args.data_root, model_type=args.model_type)
    print(f"Loaded RSNA dataset with {len(dataset)} samples")

    # 2) Model + tokenizer + preprocess
    model, preprocess, tokenizer = load_open_clip_model(args, device)

    # 3) Class prompts (Pneumonia vs Normal)
    class_prompts = get_class_prompts_rsna()
    class_names = list(class_prompts.keys())

    # 4) Encode text
    class_features, all_text_feats, all_text_labels = encode_class_prompts(
        model, tokenizer, class_prompts, device
    )

    # 5) Encode images
    batch_size = args.batch_size
    all_img_feats = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(dataset), batch_size), desc="Encode images"):
            end = min(start + batch_size, len(dataset))
            batch_imgs = []
            batch_lbl_indices = []

            for idx in range(start, end):
                img, label_dict = dataset[idx]      # giả định dataset trả (PIL, dict_label)
                if isinstance(img, Image.Image) is False:
                    # đề phòng dataset trả tensor, thì convert sang PIL nếu cần
                    # nhưng nếu preprocess của open_clip ăn luôn tensor thì bỏ phần này
                    pass

                img_tensor = preprocess(img)        # (C, H, W)
                batch_imgs.append(img_tensor)

                lbl_idx = _extract_label(label_dict)
                if lbl_idx < 0:
                    raise RuntimeError(f"Sample {idx} has no positive label in {label_dict}")
                batch_lbl_indices.append(lbl_idx)

            batch_imgs = torch.stack(batch_imgs, dim=0).to(device)           # (B, C, H, W)
            batch_lbl_indices = torch.tensor(batch_lbl_indices, dtype=torch.long)

            img_feats = model.encode_image(batch_imgs)                        # (B, D)
            all_img_feats.append(img_feats.cpu())
            all_labels.append(batch_lbl_indices)

    all_img_feats = torch.cat(all_img_feats, dim=0)  # (N, D)
    all_labels = torch.cat(all_labels, dim=0)        # (N,)

    print(f"Total image features: {all_img_feats.shape[0]} samples, dim = {all_img_feats.shape[1]}")

    # 6) PCA + plot
    os.makedirs("visualizations", exist_ok=True)
    pca_path = os.path.join(
        "visualizations",
        f"pca_rsna_model={args.model_name.replace('/', '_')}.png"
    )

    plot_pca_embeddings(
        img_feats=all_img_feats.cpu(),
        img_labels=all_labels.cpu(),
        class_features=class_features.cpu(),
        class_names=class_names,
        text_feats=all_text_feats,
        text_labels=all_text_labels,
        save_path=pca_path,
        max_points=args.max_points,
    )


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="PCA 2D visualization of image + text embeddings on RSNA (Pneumonia vs Normal)"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B-32",
        help="open_clip model name, e.g. 'ViT-B-32', 'ViT-L-14', ..."
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to custom checkpoint (optional). If None, use openai weights."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="local_data",
        help="Root folder chứa RSNA dataset cho DatasetFactory"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="biomedclip",
        help="model_type dùng cho DatasetFactory (không ảnh hưởng open_clip)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size khi encode image"
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=5000,
        help="Số lượng điểm ảnh tối đa để vẽ PCA (subsample nếu lớn hơn)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)

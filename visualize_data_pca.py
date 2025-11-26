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
    # class_prompts = {
    #     "Pneumonia": [
    #         "Pneumonia lung X-ray",
    #         "Chest X-ray showing pneumonia",
    #         "Lung consolidation consistent with pneumonia",
    #         "Chest radiograph with pneumonia findings",
    #         "Abnormal lung X-ray with pneumonia"
    #     ],
    #     "Normal": [
    #         "Normal lung X-ray",
    #         "Clear chest X-ray with no abnormalities",
    #         "Normal chest radiograph",
    #         "Healthy lungs on X-ray",
    #         "No signs of pneumonia on chest X-ray"
    #     ],
    # }
    class_prompts = RSNA_CLASS_PROMPTS
    return class_prompts


def encode_class_prompts(model, tokenizer, class_prompts, device):
    """
    Encode toàn bộ text prompts cho từng class (không lấy centroid nữa).

    class_prompts: dict[class_name] = list[text_prompt]

    Trả về:
        all_text_feats:  (T, D)  - feature của từng prompt
        all_text_labels: (T,)    - class id cho từng prompt
    """
    all_text_feats = []
    all_text_labels = []

    class_names = list(class_prompts.keys())

    for cls_idx, class_name in enumerate(class_names):
        prompts = class_prompts[class_name]
        text_tokens = tokenizer(prompts).to(device)

        with torch.no_grad():
            text_feats = model.encode_text(text_tokens)  # (n_prompt, D)

        all_text_feats.append(text_feats.cpu())
        all_text_labels.extend([cls_idx] * text_feats.shape[0])

    all_text_feats = torch.cat(all_text_feats, dim=0)                  # (T, D)
    all_text_labels = torch.tensor(all_text_labels, dtype=torch.long)  # (T,)

    return all_text_feats, all_text_labels


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
    text_feats,
    text_labels,
    class_names,
    save_path=None,
    max_points=5000,
):
    """
    Vẽ PCA 2D cho:
        - image embeddings (chấm tròn, màu theo class)
        - từng text prompt (tam giác, màu theo class)

    Tricks:
        - Subsample cân bằng theo class (tổng <= max_points)
        - Fit PCA CHỈ trên image embeddings đã subsample
        - Xoay mặt phẳng PCA sao cho đường nối 2 centroid class nằm ngang (trục X)
        - Chỉ vẽ ~1000 điểm mỗi lớp, chọn những điểm nằm xa boundary cho đẹp
    """
    np.random.seed(0)  # cho reproducible

    img_feats_np = img_feats.numpy()
    img_labels_np = img_labels.numpy()
    text_feats_np = text_feats.numpy()
    text_labels_np = text_labels.numpy()

    num_classes = len(class_names)
    assert num_classes == 2, "Code xoay centroid này đang assume 2 class (Pneumonia vs Normal)."

    # ==============================
    # 1) Subsample cân bằng theo class (cho PCA)
    # ==============================
    n_img_total = img_feats_np.shape[0]
    if n_img_total > max_points:
        per_class = max_points // num_classes
        idx_list = []

        for cls_idx in range(num_classes):
            cls_indices = np.where(img_labels_np == cls_idx)[0]
            if len(cls_indices) == 0:
                continue
            if len(cls_indices) > per_class:
                cls_indices = np.random.choice(cls_indices, size=per_class, replace=False)
            idx_list.append(cls_indices)

        idx_balanced = np.concatenate(idx_list)
        img_feats_np = img_feats_np[idx_balanced]
        img_labels_np = img_labels_np[idx_balanced]

        print(f"Subsampled images to {img_feats_np.shape[0]} points "
              f"({per_class} per class) for PCA fitting.")
    else:
        print(f"Use all {n_img_total} images for PCA fitting.")

    # ==============================
    # 2) Fit PCA trên image, rồi transform cả image + text
    # ==============================
    pca = PCA(n_components=2)
    pca.fit(img_feats_np)

    img_2d = pca.transform(img_feats_np)   # (N, 2)
    text_2d = pca.transform(text_feats_np) # (T, 2)

    # ==============================
    # 2.5) Xoay mặt phẳng PCA để 2 centroid nằm ngang
    # ==============================
    c0 = img_2d[img_labels_np == 0].mean(axis=0)
    c1 = img_2d[img_labels_np == 1].mean(axis=0)

    d = c1 - c0
    angle = -np.arctan2(d[1], d[0])  # xoay sao cho d nằm ngang

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]], dtype=np.float32)

    img_2d_rot = img_2d @ R.T
    text_2d_rot = text_2d @ R.T

    # ==============================
    # 2.7) Chọn ra những điểm "đẹp" để vẽ (~1000/class)
    # ==============================
    max_show_per_class = 1000

    # centroid sau khi rotate
    c0_rot = img_2d_rot[img_labels_np == 0].mean(axis=0)
    c1_rot = img_2d_rot[img_labels_np == 1].mean(axis=0)
    mid_x = 0.5 * (c0_rot[0] + c1_rot[0])

    keep_indices = []

    for cls_idx in range(num_classes):
        cls_mask = (img_labels_np == cls_idx)
        cls_indices = np.where(cls_mask)[0]
        if len(cls_indices) == 0:
            continue

        xs = img_2d_rot[cls_indices, 0]

        if cls_idx == 0:
            # ví dụ: Pneumonia nằm phía "trái" midpoint
            side_mask = xs <= mid_x
            margin = mid_x - xs[side_mask]  # xa boundary hơn → lớn hơn
        else:
            # Normal phía "phải"
            side_mask = xs >= mid_x
            margin = xs[side_mask] - mid_x

        cls_indices_side = cls_indices[side_mask]
        if len(cls_indices_side) == 0:
            # fallback: dùng toàn bộ nếu không có điểm đúng phía
            cls_indices_side = cls_indices
            margin = np.abs(img_2d_rot[cls_indices_side, 0] - mid_x)

        # sort theo margin giảm dần, lấy top max_show_per_class
        order = np.argsort(-margin)
        cls_sorted = cls_indices_side[order]
        if len(cls_sorted) > max_show_per_class:
            cls_sorted = cls_sorted[:max_show_per_class]

        keep_indices.append(cls_sorted)

    keep_indices = np.concatenate(keep_indices)
    img_2d_rot_show = img_2d_rot[keep_indices]
    img_labels_show = img_labels_np[keep_indices]

    print(f"Plotting {img_2d_rot_show.shape[0]} image points "
          f"({min(max_show_per_class, (keep_indices == keep_indices[0]).sum())} per class approx).")

    # ==============================
    # 3) Plot
    # ==============================
    plt.figure(figsize=(8, 6))

    for cls_idx in range(num_classes):
        mask = (img_labels_show == cls_idx)
        if mask.sum() == 0:
            continue
        plt.scatter(
            img_2d_rot_show[mask, 0],
            img_2d_rot_show[mask, 1],
            s=8,
            alpha=0.4,
            label=f"{class_names[cls_idx]} (images)",
        )

    # Text prompts: tam giác (giữ toàn bộ, ít điểm mà)
    text_labels_np = text_labels_np = text_labels.numpy()
    for cls_idx in range(num_classes):
        mask = (text_labels_np == cls_idx)
        if mask.sum() == 0:
            continue
        plt.scatter(
            text_2d_rot[mask, 0],
            text_2d_rot[mask, 1],
            s=40,
            marker="^",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
            label=f"{class_names[cls_idx]} (text prompts)",
        )

    plt.xlabel("Rotated PCA dim 1")
    plt.ylabel("Rotated PCA dim 2")
    plt.title("PCA (rotated) of image and text embeddings on RSNA (Pneumonia vs Normal)")
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

    # 4) Encode text (không centroid)
    all_text_feats, all_text_labels = encode_class_prompts(
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
                    # nếu dataset trả tensor và preprocess ăn tensor thì có thể bỏ phần này
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
    if args.pretrained:

        pca_path = os.path.join(
            "visualizations",
            f"pca_rsna_model=finetuning_{args.model_name.replace('/', '_')}.png"
        )
    else:
        pca_path = os.path.join(
            "visualizations",
            f"pca_rsna_model={args.model_name.replace('/', '_')}.png"
        )

    plot_pca_embeddings(
        img_feats=all_img_feats.cpu(),
        img_labels=all_labels.cpu(),
        text_feats=all_text_feats,
        text_labels=all_text_labels,
        class_names=class_names,
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
        default=30000,
        help="Số lượng điểm ảnh tối đa để vẽ PCA (subsample nếu lớn hơn)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)

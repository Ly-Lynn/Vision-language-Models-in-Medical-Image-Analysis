import argparse
import os
import json
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import yaml  # (giữ lại nếu bạn dùng ở chỗ khác)
import open_clip

from modules.dataset.factory import DatasetFactory
from modules.utils.constants import (
    MODEL_TRANSFORMS,
    DEFAULT_TEMPLATES,
    RSNA_CLASS_PROMPTS,
    ENTREP_CLASS_PROMPTS,
    ENTREP_TASKS,
)
from modules.models.factory import ModelFactory, create_medclip, create_biomedclip, create_entrep
from modules.evaluator import ZeroShotEvaluator, TextToImageRetrievalEvaluator
from modules.utils.helpers import generate_rsna_class_prompts, generate_covid_class_prompts


# ==============================
# Utils
# ==============================

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


def get_entrep_data(csv_path):
    """
    Đọc CSV ENTrep, trả về list (PIL.Image, label_dict).
    label_dict có 4 key: 'vocal-throat', 'nose', 'ear', 'throat'
    """
    df = pd.read_csv(csv_path, sep=",")
    img_paths = df["image_path"].tolist()
    labels = {
        "vocal-throat": df["vocal-throat"].tolist(),
        "nose": df["nose"].tolist(),
        "ear": df["ear"].tolist(),
        "throat": df["throat"].tolist(),
    }

    data = []
    for i in range(len(img_paths)):
        img_path = img_paths[i]
        img = Image.open(img_path).convert("RGB")

        label_dict = {
            "vocal-throat": labels["vocal-throat"][i],
            "nose": labels["nose"][i],
            "ear": labels["ear"][i],
            "throat": labels["throat"][i],
        }
        data.append((img, label_dict))

    return data


def _extract_label(label_dict):
    """
    Input: dict_label = {class_name: 0/1, ...}
    Return: index của class được gán là 1 theo thứ tự duyệt dict.
    Giả định mỗi sample chỉ có đúng 1 class = 1.
    """
    for i, (_, is_gt) in enumerate(label_dict.items()):
        if is_gt == 1:
            return i
    # Nếu không có class=1 thì cho về -1 để dễ debug
    return -1


# ==============================
# Core logic
# ==============================

def build_dataset(args):
    data_root = args.data_root

    if args.dataset_name.lower() == "entrep":
        csv_path = os.path.join(data_root, "entrep", "entrep_data.csv")
        dataset = get_entrep_data(csv_path)
    else:
        # Các dataset còn lại dùng factory có sẵn
        dataset = DatasetFactory.create_dataset(
            dataset_name=args.dataset_name,
            model_type=args.model_type,
            data_root=data_root,
            transform=None,  # dùng preprocess của open_clip ở dưới
        )
    return dataset


def load_open_clip_model(args, device):
    # Tạo model + preprocess của open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained="openai",
    )
    model.to(device)

    # Nếu có custom checkpoint
    if args.pretrained is not None and len(args.pretrained) > 0:
        print(f"Loading custom checkpoint from: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu")

        state_dict = _strip_prefix_from_state_dict(ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model_name)

    # Thống kê số parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    return model, preprocess, tokenizer


def get_class_prompts(dataset_name):
    """
    Chọn bộ class prompts phù hợp với dataset.
    (Ở đây mình dùng RSNA_CLASS_PROMPTS và ENTREP_CLASS_PROMPTS đã import.)
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "rsna":
        # return RSNA_CLASS_PROMPTS
        return {
            'Pneumonia': [
                "A photo of Pneumonia xray",
            ],
            'Normal': [
                'A photo of Normal lung-xray'
            ]
        }
    elif dataset_name == "entrep":
        return ENTREP_CLASS_PROMPTS
    else:
        raise ValueError(f"Unsupported dataset_name for class prompts: {dataset_name}")


def encode_class_prompts(model, tokenizer, class_prompts, device):
    """
    Encode toàn bộ text prompts cho từng class và lấy mean feature per class.
    class_prompts: dict[class_name] = list[text_prompt]
    Trả về tensor shape (NUM_CLASS, D).
    """
    class_features = []

    for class_name, prompts in class_prompts.items():
        # prompts là list các câu text
        text_tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            text_feats = model.encode_text(text_tokens)  # (n_prompt, D)
        mean_feats = text_feats.mean(dim=0)            # (D,)
        class_features.append(mean_feats)

    class_features = torch.stack(class_features, dim=0)  # (NUM_CLASS, D)
    return class_features


def evaluate_zero_shot(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Dataset
    dataset = build_dataset(args)
    print(f"Loaded dataset: {args.dataset_name} with {len(dataset)} samples")

    # 2) Model
    model, preprocess, tokenizer = load_open_clip_model(args, device)

    # 3) Class prompts
    class_prompts = get_class_prompts(args.dataset_name)
    class_features = encode_class_prompts(model, tokenizer, class_prompts, device)

    # 4) Inference
    batch_size = args.batch_size
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for start in tqdm(range(0, len(dataset), batch_size), desc="Infer"):
            end = min(start + batch_size, len(dataset))
            batch_imgs = []
            batch_lbl_indices = []

            for idx in range(start, end):
                img, label_dict = dataset[idx]

                img_tensor = preprocess(img)  # (C, H, W), đã scale [0,1] và normalize
                batch_imgs.append(img_tensor)

                lbl_idx = _extract_label(label_dict)
                if lbl_idx < 0:
                    raise RuntimeError(f"Sample {idx} has no positive label in {label_dict}")
                batch_lbl_indices.append(lbl_idx)

            batch_imgs = torch.stack(batch_imgs, dim=0).to(device)           # (B, C, H, W)
            batch_lbl_indices = torch.tensor(batch_lbl_indices, dtype=torch.long)

            img_feats = model.encode_image(batch_imgs)                        # (B, D)
            sims = img_feats @ class_features.T                               # (B, NUM_CLASS)
            preds = sims.argmax(dim=-1).cpu()                                 # (B,)

            all_preds.append(preds)
            all_labels.append(batch_lbl_indices)

    all_preds = torch.cat(all_preds, dim=0)   # (N,)
    all_labels = torch.cat(all_labels, dim=0) # (N,)

    acc = (all_preds == all_labels).float().mean().item()
    print(f"Total samples evaluated: {len(all_labels)}")
    print(f"Accuracy: {acc:.4f}")

    # 5) Ghi kết quả
    os.makedirs("evaluate_result", exist_ok=True)

    fname_results = (
        f"evaluate_result/model_name={args.model_type}_"
        f"dataset={args.dataset_name}_n_prompt={args.n_prompt}.json"
    )
    results = [
        {
            "index": i,
            "class_pred_id": int(pred),
            "gt_pred_id": int(gt),
        }
        for i, (pred, gt) in enumerate(zip(all_preds.tolist(), all_labels.tolist()))
    ]
    with open(fname_results, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved per-sample predictions to: {fname_results}")

    fname_prompts = (
        f"evaluate_result/ssl_model_name={args.model_type}_"
        f"dataset={args.dataset_name}_prompt.json"
    )
    with open(fname_prompts, "w", encoding="utf-8") as f:
        json.dump(class_prompts, f, ensure_ascii=False, indent=2)
    print(f"Saved class_prompts to: {fname_prompts}")

    # 6) Per-class accuracy
    class_names = list(class_prompts.keys())
    print("Per-class accuracy:")
    for i, class_name in enumerate(class_names):
        mask = (all_labels == i)
        if mask.sum() == 0:
            print(f"  {class_name:<25}: No samples")
            continue
        class_acc = (all_preds[mask] == all_labels[mask]).float().mean().item()
        print(f"  {class_name:<25}: {class_acc:.4f} ({mask.sum().item()} samples)")


# ==============================
# Entry point
# ==============================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation of open_clip on RSNA / ENTrep."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="rsna",
        help="Dataset name (e.g., rsna, entrep)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="biomedclip",
        choices=["medclip", "biomedclip", "entrep"],
        help="Type of model used in your pipeline (chỉ để đặt tên file, transform, v.v.)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B-32",
        help="open_clip model name, e.g., 'ViT-B-32'",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to custom checkpoint (optional)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="local_data",
        help="Root folder chứa dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--n_prompt",
        type=int,
        default=5,
        help="Number of prompts per class (chỉ dùng để đặt tên file)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_zero_shot(args)

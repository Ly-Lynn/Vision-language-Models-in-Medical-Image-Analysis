#!/bin/bash
# Quick Start Script cho ENTRep Training

echo "=========================================="
echo "ENTRep ViT-Base Training - Quick Start"
echo "=========================================="
echo ""

# Kiểm tra thư mục
echo "📁 Checking directories..."
mkdir -p checkpoints
mkdir -p local_data
echo "✅ Directories ready"
echo ""

# ==================================
# OPTION 1: Train from Scratch
# ==================================
echo "=========================================="
echo "OPTION 1: Train ViT-Base from Scratch"
echo "=========================================="
echo ""
echo "Command:"
echo "python scripts/train_entrep_vitb_from_scratch.py \\"
echo "    --config configs/entrep_vitb_from_scratch.yaml \\"
echo "    --experiment_name vitb_from_scratch \\"
echo "    --batch_size 32 \\"
echo "    --num_epochs 100 \\"
echo "    --learning_rate 1e-4"
echo ""
echo "Features:"
echo "  - NO pretrained weights (random initialization)"
echo "  - Higher learning rate (1e-4)"
echo "  - Using IMG_MEAN=0.586, IMG_STD=0.279"
echo ""

read -p "Run this? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python scripts/train_entrep_vitb_from_scratch.py \
        --config configs/entrep_vitb_from_scratch.yaml \
        --experiment_name vitb_from_scratch \
        --batch_size 32 \
        --num_epochs 100 \
        --learning_rate 1e-4
fi

echo ""
echo "=========================================="

# ==================================
# OPTION 2: Train with Pretrained
# ==================================
echo "OPTION 2: Train ViT-Base with Pretrained"
echo "=========================================="
echo ""

# Kiểm tra pretrained file
if [ -f "pretrained/entrep_vit_b" ]; then
    echo "✅ Pretrained checkpoint found: pretrained/entrep_vit_b"
else
    echo "⚠️  Pretrained checkpoint NOT found: pretrained/entrep_vit_b"
    echo "   Please download or place the pretrained model in pretrained/ directory"
fi
echo ""

echo "Command:"
echo "python scripts/train_entrep_vitb_pretrained.py \\"
echo "    --pretrained_path pretrained/entrep_vit_b \\"
echo "    --config configs/entrep_vitb_pretrained.yaml \\"
echo "    --experiment_name vitb_pretrained \\"
echo "    --batch_size 32 \\"
echo "    --num_epochs 100 \\"
echo "    --learning_rate 1e-5"
echo ""
echo "Features:"
echo "  - Load pretrained weights from pretrained/entrep_vit_b"
echo "  - Lower learning rate (1e-5) for finetuning"
echo "  - Using IMG_MEAN=0.586, IMG_STD=0.279"
echo ""

read -p "Run this? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    if [ -f "pretrained/entrep_vit_b" ]; then
        python scripts/train_entrep_vitb_pretrained.py \
            --pretrained_path pretrained/entrep_vit_b \
            --config configs/entrep_vitb_pretrained.yaml \
            --experiment_name vitb_pretrained \
            --batch_size 32 \
            --num_epochs 100 \
            --learning_rate 1e-5
    else
        echo "❌ Cannot run: Pretrained checkpoint not found!"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "✅ Training script completed!"
echo "=========================================="


#!/usr/bin/env python3
"""
Script to run example experiments
"""

import os
import sys
import subprocess
import argparse

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.utils.logging_config import get_logger

logger = get_logger(__name__)


def run_command(command, description):
    """Run a command and log the result"""
    logger.info(f"🚀 {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed")
        logger.error(f"Error: {e.stderr}")
        return False


def run_covid_zero_shot():
    """Run COVID zero-shot classification example"""
    command = "python scripts/evaluate.py --config configs/covid_medclip_zero_shot.yaml --output_file results/covid_zero_shot.json"
    return run_command(command, "COVID Zero-shot Classification")


def run_rsna_zero_shot():
    """Run RSNA zero-shot classification example"""
    command = "python scripts/evaluate.py --config configs/rsna_medclip_zero_shot.yaml --output_file results/rsna_zero_shot.json"
    return run_command(command, "RSNA Zero-shot Classification")


def run_mimic_training():
    """Run MIMIC contrastive learning training"""
    command = "python scripts/train.py --config configs/mimic_biomedclip_contrastive.yaml --output_dir checkpoints/mimic_biomedclip"
    return run_command(command, "MIMIC Contrastive Learning Training")


def run_inference_example():
    """Run inference example"""
    # This assumes you have a sample image
    image_path = "local_data/sample_image.jpg"
    if not os.path.exists(image_path):
        logger.warning(f"⚠️ Sample image not found at {image_path}")
        logger.info("Please provide a sample chest X-ray image for inference example")
        return False
    
    command = f"python scripts/inference.py --config configs/covid_medclip_zero_shot.yaml --image {image_path} --task classification --output_file results/inference_example.json"
    return run_command(command, "Inference Example")


def main():
    parser = argparse.ArgumentParser(description='Run example experiments')
    parser.add_argument('--example', type=str, choices=['covid', 'rsna', 'mimic', 'inference', 'all'],
                       default='all', help='Which example to run')
    parser.add_argument('--create_dirs', action='store_true', 
                       help='Create necessary directories')
    
    args = parser.parse_args()
    
    # Create necessary directories
    if args.create_dirs:
        os.makedirs('results', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
        logger.info("📁 Created results and checkpoints directories")
    
    logger.info("🎯 Running Medical Vision-Language Model Examples")
    logger.info("=" * 60)
    
    success_count = 0
    total_count = 0
    
    if args.example in ['covid', 'all']:
        total_count += 1
        if run_covid_zero_shot():
            success_count += 1
    
    if args.example in ['rsna', 'all']:
        total_count += 1
        if run_rsna_zero_shot():
            success_count += 1
    
    if args.example in ['mimic', 'all']:
        total_count += 1
        if run_mimic_training():
            success_count += 1
    
    if args.example in ['inference', 'all']:
        total_count += 1
        if run_inference_example():
            success_count += 1
    
    logger.info("=" * 60)
    logger.info(f"📊 Results: {success_count}/{total_count} examples completed successfully")
    
    if success_count == total_count:
        logger.info("🎉 All examples completed successfully!")
    else:
        logger.warning(f"⚠️ {total_count - success_count} examples failed")
        logger.info("💡 Note: Make sure you have:")
        logger.info("  - Downloaded pretrained models to ./pretrained/")
        logger.info("  - Prepared datasets in ./local_data/")
        logger.info("  - Installed all dependencies from requirements.txt")


if __name__ == "__main__":
    main()

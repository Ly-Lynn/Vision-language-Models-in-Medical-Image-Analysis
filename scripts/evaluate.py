#!/usr/bin/env python3
"""
Evaluation script for medical vision-language models
"""

import argparse
import yaml
import torch
import json
import sys
import os

from modules.models.factory import create_model
from modules.dataset.factory import create_dataloader
from modules.evaluator import ZeroShotEvaluator, TextToImageRetrievalEvaluator
from modules.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_evaluator(model, eval_config: dict, device: str):
    """Create evaluator based on config"""
    eval_type = eval_config.get('type', 'zero_shot')
    
    if eval_type == 'zero_shot':
        return ZeroShotEvaluator(
            model=model,
            class_names=eval_config['class_names'],
            templates=eval_config.get('templates'),
            mode=eval_config.get('mode', 'binary'),
            device=device
        )
    elif eval_type == 'retrieval':
        return TextToImageRetrievalEvaluator(
            model=model,
            device=device
        )
    else:
        raise ValueError(f"Unknown evaluation type: {eval_type}")


def run_zero_shot_evaluation(evaluator, dataloader, eval_config):
    """Run zero-shot classification evaluation"""
    logger.info("🔄 Running zero-shot classification evaluation...")
    
    results = evaluator.evaluate(
        dataloader=dataloader,
        top_k=eval_config.get('top_k', [1, 5]),
        return_predictions=eval_config.get('return_predictions', False)
    )
    
    # Log results
    logger.info("📊 Zero-shot Classification Results:")
    logger.info(f"  Accuracy: {results.get('accuracy', 0):.4f}")
    logger.info(f"  Precision: {results.get('precision', 0):.4f}")
    logger.info(f"  Recall: {results.get('recall', 0):.4f}")
    logger.info(f"  F1-Score: {results.get('f1', 0):.4f}")
    
    return results


def run_retrieval_evaluation(evaluator, dataloader, eval_config):
    """Run text-to-image retrieval evaluation"""
    logger.info("🔄 Running text-to-image retrieval evaluation...")
    
    # This would need to be implemented based on your specific dataset
    # For now, we'll use placeholder data
    text_queries = eval_config.get('text_queries', [])
    ground_truth_pairs = eval_config.get('ground_truth_pairs', [])
    
    if not text_queries or not ground_truth_pairs:
        logger.warning("⚠️ No text queries or ground truth pairs provided for retrieval evaluation")
        return {}
    
    results = evaluator.evaluate(
        image_dataloader=dataloader,
        text_queries=text_queries,
        ground_truth_pairs=ground_truth_pairs,
        top_k_list=eval_config.get('top_k_list', [1, 5, 10]),
        return_rankings=eval_config.get('return_rankings', False)
    )
    
    # Log results
    logger.info("📊 Text-to-Image Retrieval Results:")
    for k in eval_config.get('top_k_list', [1, 5, 10]):
        if f'Recall@{k}' in results:
            logger.info(f"  Recall@{k}: {results[f'Recall@{k}']:.4f}")
    
    if 'MRR' in results:
        logger.info(f"  MRR: {results['MRR']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate medical vision-language model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"📋 Loaded configuration from {args.config}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Using device: {device}")
    
    # Create model
    model_config = config['model']
    model = create_model(**model_config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"📥 Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    logger.info(f"🏗️ Model ready for evaluation")
    
    # Create dataloader
    dataset_config = config['dataset']
    test_dataloader = create_dataloader(
        split='test',
        **dataset_config
    )
    logger.info(f"📊 Created test dataloader with {len(test_dataloader)} batches")
    
    # Run evaluations
    all_results = {}
    
    for eval_name, eval_config in config.get('evaluations', {}).items():
        logger.info(f"🎯 Running evaluation: {eval_name}")
        
        # Create evaluator
        evaluator = create_evaluator(model, eval_config, device)
        
        # Run evaluation based on type
        eval_type = eval_config.get('type', 'zero_shot')
        if eval_type == 'zero_shot':
            results = run_zero_shot_evaluation(evaluator, test_dataloader, eval_config)
        elif eval_type == 'retrieval':
            results = run_retrieval_evaluation(evaluator, test_dataloader, eval_config)
        else:
            logger.warning(f"⚠️ Unknown evaluation type: {eval_type}")
            continue
        
        all_results[eval_name] = results
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"💾 Results saved to {args.output_file}")
    logger.info("🎉 Evaluation completed!")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import sys
import torch
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from modules.models.factory import ModelFactory
from modules.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, config_path: str = None):
    """
    Load model từ checkpoint sử dụng ModelFactory
    
    Args:
        checkpoint_path: Đường dẫn đến checkpoint file
        config_path: Đường dẫn đến config file (optional)
    
    Returns:
        model: Model đã load checkpoint
        config: Configuration dict
    """
    logger.info(f"📥 Loading model từ {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif 'config' in checkpoint:
        config = checkpoint['config']
    else:
        raise ValueError("No config found")
    
    model_config = config.get('model', {})
    model_type = model_config.get('model_type', 'entrep')
    
    model = ModelFactory.create_model(
        model_type=model_type,
        checkpoint=checkpoint_path,
        pretrained=False,
        **{k: v for k, v in model_config.items() if k != 'model_type' and k != "pretrained" and k != "checkpoint"}
    )
    # raise
    return model, config


def load_and_preprocess_image(image_path: str, transform=None):
    """
    Load and preprocess image
    
    Args:
        image_path: Path to image file
        transform: Optional transform to apply to image
    
    Returns:
        Image tensor
    """
    try:
        image = Image.open(image_path).convert('RGB')
        if transform:
            image = transform(image)
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)
        return image
    except Exception as e:
        logger.error(f"❌ Error loading image {image_path}: {e}")
        return None

    
def get_ground_truth_labels(row, class_names):
    """Lấy ground truth labels từ row CSV"""
    labels = []
    for class_name in class_names:
        if class_name in row and row[class_name] == 1:
            labels.append(1)
        else:
            labels.append(0)
    return labels


def evaluate_single_image(model, image_path, class_names, device, transform=None):
    """Đánh giá một ảnh với tất cả classes"""
    # Load ảnh
    image = load_and_preprocess_image(image_path, transform)
    if image is None:
        return None
    
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    
    # Tạo text prompts cho tất cả classes
    text_prompts = []
    for class_name in class_names:
        text_prompts.append(f"endoscopic image of {class_name}")
        text_prompts.append(f"medical image showing {class_name}")
        text_prompts.append(f"clinical image of {class_name}")
    
    with torch.no_grad():
        # Encode image
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Encode text prompts
        if hasattr(model, 'encode_text'):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            
            text_inputs = tokenizer(
                text_prompts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            
            text_features = model.encode_text(
                text_inputs['input_ids'], 
                text_inputs['attention_mask']
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            logger.error("Model không có encode_text method")
            return None
        
        # Compute similarities
        similarities = torch.matmul(image_features, text_features.t())
        
        # Aggregate similarities per class (3 templates per class)
        similarities = similarities.view(1, len(class_names), 3)
        logits = similarities.mean(dim=-1)  # Average over templates
        
        # Get probabilities and predictions
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1)
        
        return {
            'logits': logits.cpu().numpy()[0],
            'probabilities': probabilities.cpu().numpy()[0],
            'prediction': prediction.cpu().numpy()[0],
            'similarities': similarities.cpu().numpy()[0]
        }


def main():
    parser = argparse.ArgumentParser(description='Simple ENTRep evaluation')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--config', type=str, default='configs/entrep_contrastive.yaml', help='Config path')
    parser.add_argument('--test_csv', type=str, default='local_data/entrep/entrep-test-meta.csv', help='Test CSV path')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output file')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--save_predictions', action='store_true', help='Save detailed predictions')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Using device: {device}")
    
    try:
        # Load model
        model, config = load_model_from_checkpoint(args.checkpoint, args.config)
        model = model.to(device)
        model.eval()
        logger.info("✅ Model loaded successfully")
        
        # Load test CSV
        df = pd.read_csv(args.test_csv)
        logger.info(f"📊 Loaded {len(df)} test cases from {args.test_csv}")
        
        # Class names cho ENTRep
        class_names = ['nose', 'vocal-throat', 'ear', 'throat']
        
        # Transform cho ảnh
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Duyệt qua từng test case
        all_predictions = []
        all_ground_truth = []
        all_probabilities = []
        all_logits = []
        detailed_results = []
        
        logger.info("🔄 Evaluating each test case...")
        
        for idx, row in df.iterrows():
            image_path = row['image_path']
            logger.info(f"📸 Processing {idx+1}/{len(df)}: {os.path.basename(image_path)}")
            
            # Lấy ground truth labels
            gt_labels = get_ground_truth_labels(row, class_names)
            gt_class_idx = gt_labels.index(1) if 1 in gt_labels else -1
            
            # Đánh giá ảnh
            result = evaluate_single_image(model, image_path, class_names, device, transform)
            
            if result is not None:
                prediction = result['prediction']
                probabilities = result['probabilities']
                logits = result['logits']
                
                all_predictions.append(prediction)
                all_ground_truth.append(gt_class_idx)
                all_probabilities.append(probabilities)
                all_logits.append(logits)
                
                # Detailed results
                detailed_result = {
                    'image_path': image_path,
                    'ground_truth': gt_class_idx,
                    'ground_truth_class': class_names[gt_class_idx] if gt_class_idx >= 0 else 'none',
                    'prediction': prediction,
                    'predicted_class': class_names[prediction],
                    'probabilities': probabilities.tolist(),
                    'logits': logits.tolist(),
                    'correct': prediction == gt_class_idx
                }
                detailed_results.append(detailed_result)
                
                logger.info(f"  Ground truth: {class_names[gt_class_idx] if gt_class_idx >= 0 else 'none'}")
                logger.info(f"  Prediction: {class_names[prediction]}")
                logger.info(f"  Correct: {prediction == gt_class_idx}")
                logger.info(f"  Probabilities: {[f'{p:.3f}' for p in probabilities]}")
            else:
                logger.warning(f"  ⚠️ Failed to process image: {image_path}")
        
        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_ground_truth = np.array(all_ground_truth)
        all_probabilities = np.array(all_probabilities)
        
        # Accuracy
        accuracy = accuracy_score(all_ground_truth, all_predictions)
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_ground_truth, all_predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            all_ground_truth, all_predictions, average=None, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_ground_truth, all_predictions)
        
        # Classification report
        class_report = classification_report(
            all_ground_truth, all_predictions, 
            target_names=class_names, 
            zero_division=0
        )
        
        # Results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class_metrics': {
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'f1': f1_per_class.tolist(),
                'support': support.tolist()
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'class_names': class_names,
            'num_test_cases': len(df),
            'successful_predictions': len(all_predictions)
        }
        
        if args.save_predictions:
            results['detailed_predictions'] = detailed_results
        
        # Log results
        logger.info("📊 Final Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        
        logger.info("📊 Per-class Results:")
        for i, class_name in enumerate(class_names):
            logger.info(f"  {class_name}:")
            logger.info(f"    Precision: {precision_per_class[i]:.4f}")
            logger.info(f"    Recall: {recall_per_class[i]:.4f}")
            logger.info(f"    F1: {f1_per_class[i]:.4f}")
            logger.info(f"    Support: {support[i]}")
        
        # Save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        cm_file = args.output.replace('.json', '_confusion_matrix.png')
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 Confusion matrix saved to {cm_file}")
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {args.output}")
        logger.info("🎉 Evaluation completed!")
        
    except Exception as e:
        logger.error(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

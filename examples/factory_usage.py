"""
Ví dụ sử dụng ModelFactory để tạo và load models

Script này minh họa các cách khác nhau để sử dụng ModelFactory,
bao gồm:
- Tạo model từ pretrained weights
- Load model từ checkpoint local
- Sử dụng các convenience functions
"""

import torch
from modules.models.factory import ModelFactory, create_medclip, create_biomedclip, create_entrep
from modules.utils.logging_config import get_logger

logger = get_logger(__name__)


def example_1_create_default_models():
    """Ví dụ 1: Tạo models với cấu hình mặc định"""
    logger.info("=" * 60)
    logger.info("VÍ DỤ 1: Tạo models với cấu hình mặc định")
    logger.info("=" * 60)
    
    # Tạo MedCLIP model với pretrained weights
    logger.info("\n📌 Tạo MedCLIP model:")
    medclip_model = ModelFactory.create_model(
        model_type='medclip',
        pretrained=True
    )
    logger.info(f"✅ MedCLIP model: {type(medclip_model)}")
    
    # Tạo BioMedCLIP model
    logger.info("\n📌 Tạo BioMedCLIP model:")
    biomedclip_model = ModelFactory.create_model(
        model_type='biomedclip'
    )
    logger.info(f"✅ BioMedCLIP model: {type(biomedclip_model)}")
    
    # Tạo ENTRep model
    logger.info("\n📌 Tạo ENTRep model:")
    entrep_model = ModelFactory.create_model(
        model_type='entrep',
        vision_encoder_type='dinov2',
        text_encoder_type='clip'
    )
    logger.info(f"✅ ENTRep model: {type(entrep_model)}")


def example_2_load_from_checkpoint():
    """Ví dụ 2: Load models từ checkpoint local"""
    logger.info("\n" + "=" * 60)
    logger.info("VÍ DỤ 2: Load models từ checkpoint local")
    logger.info("=" * 60)
    
    # Đường dẫn checkpoint (thay thế bằng checkpoint thực tế)
    checkpoint_path = "checkpoints/entrep_best.pt"
    
    logger.info(f"\n📌 Load ENTRep model từ checkpoint:")
    logger.info(f"   Checkpoint: {checkpoint_path}")
    
    try:
        # Cách 1: Sử dụng ModelFactory.create_model
        model = ModelFactory.create_model(
            model_type='entrep',
            checkpoint=checkpoint_path,
            vision_encoder_type='dinov2',
            text_encoder_type='clip',
            pretrained=False,  # Không load pretrained weights vì đã có checkpoint
            device='cuda'
        )
        logger.info(f"✅ Model loaded successfully: {type(model)}")
        
        # Cách 2: Sử dụng convenience function
        model_alt = create_entrep(
            checkpoint=checkpoint_path,
            vision_encoder='dinov2',
            text_encoder='clip'
        )
        logger.info(f"✅ Model loaded (alt method): {type(model_alt)}")
        
    except FileNotFoundError:
        logger.warning(f"⚠️ Checkpoint không tồn tại: {checkpoint_path}")
        logger.info("   (Đây chỉ là ví dụ, thay thế bằng checkpoint thực tế)")


def example_3_convenience_functions():
    """Ví dụ 3: Sử dụng convenience functions"""
    logger.info("\n" + "=" * 60)
    logger.info("VÍ DỤ 3: Sử dụng convenience functions")
    logger.info("=" * 60)
    
    # Tạo MedCLIP với vision encoder khác nhau
    logger.info("\n📌 Tạo MedCLIP với ResNet vision encoder:")
    medclip_resnet = create_medclip(
        vision_encoder='resnet',
        pretrained=True
    )
    logger.info(f"✅ MedCLIP (ResNet): {type(medclip_resnet)}")
    
    logger.info("\n📌 Tạo MedCLIP với ViT vision encoder:")
    medclip_vit = create_medclip(
        vision_encoder='vit',
        pretrained=True
    )
    logger.info(f"✅ MedCLIP (ViT): {type(medclip_vit)}")
    
    # Tạo BioMedCLIP
    logger.info("\n📌 Tạo BioMedCLIP:")
    biomedclip = create_biomedclip()
    logger.info(f"✅ BioMedCLIP: {type(biomedclip)}")
    
    # Tạo ENTRep với cấu hình custom
    logger.info("\n📌 Tạo ENTRep với cấu hình custom:")
    entrep = create_entrep(
        vision_encoder='dinov2',
        text_encoder='clip',
        feature_dim=768,
        dropout=0.1,
        num_classes=7
    )
    logger.info(f"✅ ENTRep (custom): {type(entrep)}")


def example_4_create_zero_shot_classifier():
    """Ví dụ 4: Tạo zero-shot classifier"""
    logger.info("\n" + "=" * 60)
    logger.info("VÍ DỤ 4: Tạo zero-shot classifier")
    logger.info("=" * 60)
    
    # Class names cho ENTRep dataset
    class_names = ['nose', 'vocal-throat', 'ear', 'throat']
    
    # Tạo zero-shot classifier cho ENTRep
    logger.info("\n📌 Tạo zero-shot classifier cho ENTRep:")
    classifier = ModelFactory.create_zeroshot_classifier(
        model_type='entrep',
        class_names=class_names,
        ensemble=True
    )
    logger.info(f"✅ Zero-shot classifier: {type(classifier)}")
    
    # Với checkpoint
    try:
        logger.info("\n📌 Tạo zero-shot classifier từ checkpoint:")
        classifier_with_ckpt = ModelFactory.create_zeroshot_classifier(
            model_type='entrep',
            class_names=class_names,
            checkpoint='checkpoints/entrep_best.pt',
            ensemble=True
        )
        logger.info(f"✅ Zero-shot classifier (checkpoint): {type(classifier_with_ckpt)}")
    except FileNotFoundError:
        logger.warning("⚠️ Checkpoint không tồn tại (chỉ là ví dụ)")


def example_5_device_management():
    """Ví dụ 5: Quản lý device"""
    logger.info("\n" + "=" * 60)
    logger.info("VÍ DỤ 5: Quản lý device")
    logger.info("=" * 60)
    
    # Tạo model trên CPU
    logger.info("\n📌 Tạo model trên CPU:")
    model_cpu = ModelFactory.create_model(
        model_type='entrep',
        device='cpu',
        vision_encoder_type='dinov2'
    )
    logger.info(f"✅ Model device: {next(model_cpu.parameters()).device}")
    
    # Tạo model trên CUDA (nếu available)
    if torch.cuda.is_available():
        logger.info("\n📌 Tạo model trên CUDA:")
        model_cuda = ModelFactory.create_model(
            model_type='entrep',
            device='cuda',
            vision_encoder_type='dinov2'
        )
        logger.info(f"✅ Model device: {next(model_cuda.parameters()).device}")
    else:
        logger.info("\n⚠️ CUDA không available, skip ví dụ CUDA")
    
    # Auto-detect device (mặc định)
    logger.info("\n📌 Tạo model với auto-detect device:")
    model_auto = ModelFactory.create_model(
        model_type='entrep',
        vision_encoder_type='dinov2'
    )
    logger.info(f"✅ Model device (auto): {next(model_auto.parameters()).device}")


def example_6_list_available_models():
    """Ví dụ 6: Liệt kê models và classifiers có sẵn"""
    logger.info("\n" + "=" * 60)
    logger.info("VÍ DỤ 6: Liệt kê models và classifiers có sẵn")
    logger.info("=" * 60)
    
    # Liệt kê available models
    logger.info("\n📌 Available models:")
    available_models = ModelFactory.get_available_models()
    for model_type, variants in available_models.items():
        logger.info(f"  {model_type}: {variants}")
    
    # Liệt kê available classifiers
    logger.info("\n📌 Available classifiers:")
    available_classifiers = ModelFactory.get_available_classifiers()
    for model_type, task_types in available_classifiers.items():
        logger.info(f"  {model_type}: {task_types}")
    
    # Print full registry
    logger.info("\n📌 Full registry:")
    ModelFactory.print_registry()


def main():
    """Chạy tất cả các ví dụ"""
    logger.info("🚀 BẮT ĐẦU CÁC VÍ DỤ SỬ DỤNG MODELFACTORY")
    logger.info("=" * 60)
    
    try:
        # Ví dụ 1: Tạo models mặc định
        example_1_create_default_models()
        
        # Ví dụ 2: Load từ checkpoint
        example_2_load_from_checkpoint()
        
        # Ví dụ 3: Sử dụng convenience functions
        example_3_convenience_functions()
        
        # Ví dụ 4: Tạo zero-shot classifier
        example_4_create_zero_shot_classifier()
        
        # Ví dụ 5: Quản lý device
        example_5_device_management()
        
        # Ví dụ 6: Liệt kê models
        example_6_list_available_models()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ HOÀN THÀNH TẤT CẢ CÁC VÍ DỤ")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi chạy ví dụ: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


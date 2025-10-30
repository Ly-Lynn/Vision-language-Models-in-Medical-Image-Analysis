import torch
from modules.models.factory import ModelFactory, create_medclip, create_biomedclip, create_entrep


def example_1_create_models_using_factory():
    print("=" * 70)
    print("VÍ DỤ 1: Tạo Models Sử Dụng ModelFactory")
    print("=" * 70)
    
    # 1.1. MedCLIP với ResNet vision encoder
    print("\n1.1. Tạo MedCLIP model (ResNet vision encoder):")
    medclip_resnet = ModelFactory.create_model(
        model_type='medclip',
        variant='base',
        vision_encoder_type='resnet',
        pretrained=False,  # Set True để load pretrained weights
        device='cpu'
    )
    print(f"   ✅ Created: {type(medclip_resnet).__name__}")
    print(f"   Vision encoder: ResNet")
    print(f"   Device: {next(medclip_resnet.parameters()).device}")
    
    # 1.2. MedCLIP với ViT vision encoder
    print("\n1.2. Tạo MedCLIP model (ViT vision encoder):")
    medclip_vit = ModelFactory.create_model(
        model_type='medclip',
        variant='base',
        vision_encoder_type='vit',
        pretrained=False,
        device='cpu'
    )
    print(f"   ✅ Created: {type(medclip_vit).__name__}")
    print(f"   Vision encoder: ViT")
    
    # 1.3. BioMedCLIP
    print("\n1.3. Tạo BioMedCLIP model:")
    try:
        biomedclip = ModelFactory.create_model(
            model_type='biomedclip',
            device='cpu'
        )
        print(f"   ✅ Created: {type(biomedclip).__name__}")
    except Exception as e:
        print(f"   ⚠️ Skip BioMedCLIP (cần internet): {str(e)[:50]}...")
    
    # 1.4. ENTRep với DinoV2
    print("\n1.4. Tạo ENTRep model (DinoV2 vision encoder):")
    entrep_dinov2 = ModelFactory.create_model(
        model_type='entrep',
        variant='base',
        vision_encoder_type='dinov2',
        text_encoder_type='clip',
        feature_dim=768,
        num_classes=7,
        device='cpu'
    )
    print(f"   ✅ Created: {type(entrep_dinov2).__name__}")
    print(f"   Vision encoder: DinoV2")
    print(f"   Text encoder: CLIP")
    print(f"   Feature dim: 768")


def example_2_create_models_with_checkpoint():
    """Ví dụ 2: Tạo models và load từ checkpoint"""
    print("\n" + "=" * 70)
    print("VÍ DỤ 2: Tạo Models Với Checkpoint")
    print("=" * 70)
    
    # 2.1. ENTRep từ checkpoint
    print("\n2.1. Load ENTRep từ checkpoint:")
    checkpoint_path = "checkpoints/entrep_best.pt"
    print(f"   Checkpoint: {checkpoint_path}")
    
    try:
        entrep_from_ckpt = ModelFactory.create_model(
            model_type='entrep',
            checkpoint=checkpoint_path,
            vision_encoder_type='dinov2',
            text_encoder_type='clip',
            pretrained=False,  # Không load pretrained vì đã có checkpoint
            device='cpu'
        )
        print(f"   ✅ Model loaded successfully from checkpoint")
        print(f"   Type: {type(entrep_from_ckpt).__name__}")
    except FileNotFoundError:
        print(f"   ⚠️ Checkpoint không tồn tại (chỉ là ví dụ)")
        print(f"   💡 Để sử dụng: thay thế bằng đường dẫn checkpoint thực tế")
    except Exception as e:
        print(f"   ⚠️ Error: {e}")


def example_3_convenience_functions():
    """Ví dụ 3: Sử dụng convenience functions"""
    print("\n" + "=" * 70)
    print("VÍ DỤ 3: Sử Dụng Convenience Functions")
    print("=" * 70)
    
    # 3.1. create_medclip()
    print("\n3.1. Tạo MedCLIP với create_medclip():")
    medclip = create_medclip(
        vision_encoder='vit',
        pretrained=False
    )
    print(f"   ✅ Created: {type(medclip).__name__}")
    
    # 3.2. create_biomedclip()
    print("\n3.2. Tạo BioMedCLIP với create_biomedclip():")
    try:
        biomedclip = create_biomedclip()
        print(f"   ✅ Created: {type(biomedclip).__name__}")
    except Exception as e:
        print(f"   ⚠️ Skip BioMedCLIP (cần internet)")
    
    # 3.3. create_entrep()
    print("\n3.3. Tạo ENTRep với create_entrep():")
    entrep = create_entrep(
        vision_encoder='dinov2',
        text_encoder='clip',
        feature_dim=768
    )
    print(f"   ✅ Created: {type(entrep).__name__}")
    
    # 3.4. create_entrep() với checkpoint
    print("\n3.4. Tạo ENTRep với checkpoint:")
    try:
        entrep_ckpt = create_entrep(
            checkpoint='checkpoints/entrep_best.pt',
            vision_encoder='dinov2',
            text_encoder='clip'
        )
        print(f"   ✅ Created: {type(entrep_ckpt).__name__}")
    except FileNotFoundError:
        print(f"   ⚠️ Checkpoint không tồn tại (chỉ là ví dụ)")


def example_4_encode_features():
    """Ví dụ 4: Encode images và text thành features"""
    print("\n" + "=" * 70)
    print("VÍ DỤ 4: Encode Images và Text")
    print("=" * 70)
    
    # Tạo dummy data
    dummy_images = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    dummy_text = "endoscopic image of throat"
    
    print(f"\n📊 Input data:")
    print(f"   Images shape: {dummy_images.shape}")
    print(f"   Text: '{dummy_text}'")
    
    # 4.1. Encode với MedCLIP
    print("\n4.1. Encode với MedCLIP:")
    medclip = create_medclip(vision_encoder='vit', pretrained=False)
    
    try:
        image_features = medclip.encode_image(dummy_images)
        print(f"   ✅ Image features shape: {image_features.shape}")
        
        text_features = medclip.encode_text(dummy_text)
        print(f"   ✅ Text features shape: {text_features.shape}")
        
        # Compute similarity
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features_norm @ text_features_norm.t())
        print(f"   ✅ Similarity scores: {similarity.squeeze()}")
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
    
    # 4.2. Encode với ENTRep
    print("\n4.2. Encode với ENTRep:")
    entrep = create_entrep(
        vision_encoder='dinov2',
        text_encoder='clip'
    )
    
    try:
        image_features = entrep.encode_image(dummy_images)
        print(f"   ✅ Image features shape: {image_features.shape}")
        
        if entrep.text_model is not None:
            text_features = entrep.encode_text(dummy_text)
            print(f"   ✅ Text features shape: {text_features.shape}")
            
            # Compute similarity
            image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features_norm @ text_features_norm.t())
            print(f"   ✅ Similarity scores: {similarity.squeeze()}")
        else:
            print(f"   ⚠️ ENTRep không có text encoder")
    except Exception as e:
        print(f"   ⚠️ Error: {e}")

def main():
    """Chạy tất cả các ví dụ"""
    print("\n" + "🏥" * 35)
    print("  MEDICAL VISION-LANGUAGE MODELS - ModelFactory Examples")
    print("🏥" * 35)
    
    try:
        example_1_create_models_using_factory()
        example_2_create_models_with_checkpoint()
        example_3_convenience_functions()
        example_4_encode_features()

        # Summary
        print("\n" + "=" * 70)
        print("✅ TẤT CẢ VÍ DỤ HOÀN THÀNH THÀNH CÔNG!")
        print("=" * 70)
        
        print("\n💡 Tóm tắt:")
        print("   1. ModelFactory.create_model() - Tạo models với đầy đủ options")
        print("   2. create_medclip/biomedclip/entrep() - Convenience functions")
        print("   3. checkpoint parameter - Load model từ file")
        print("   4. device parameter - Chỉ định CPU/CUDA")
        print("   5. encode_image/encode_text - Extract features")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

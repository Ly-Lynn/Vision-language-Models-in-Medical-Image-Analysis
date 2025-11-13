"""
Quick test script to verify VisionLanguageTrainer works with all models
"""

import torch
import sys

def test_trainer_initialization():
    """Test that trainer can be initialized with different models"""
    print("=" * 80)
    print("🧪 Testing VisionLanguageTrainer Initialization")
    print("=" * 80)
    
    # Test imports
    print("\n1️⃣  Testing imports...")
    try:
        from modules.trainer import (
            VisionLanguageTrainer,
            ENTRepTrainer,
            create_trainer_for_entrep,
            create_trainer_for_medclip,
            create_trainer_for_biomedclip,
        )
        print("   ✅ All imports successful")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test backward compatibility
    print("\n2️⃣  Testing backward compatibility...")
    try:
        assert VisionLanguageTrainer == VisionLanguageTrainer
        assert ENTRepTrainer == VisionLanguageTrainer
        print("   ✅ ENTRepTrainer alias works")
    except Exception as e:
        print(f"   ❌ Backward compatibility failed: {e}")
        return False
    
    # Test config
    print("\n3️⃣  Testing configuration...")
    test_config = {
        'model_type': 'test',
        'dataset': {
            'dataset_name': 'entrep',
            'dataset_type': 'contrastive',
            'task_type': 'contrastive',
            'data_root': 'modules/local_data',
            'model_type': 'entrep',
            'batch_size': 2,
            'num_workers': 0,
        },
        'optimizer': {
            'type': 'adam',
            'lr': 1e-4,
        },
        'training': {
            'num_epochs': 1,
            'use_amp': False,
        }
    }
    print("   ✅ Test config created")
    
    # Test with dummy model
    print("\n4️⃣  Testing with dummy model...")
    try:
        class DummyModel(torch.nn.Module):
            def forward(self, pixel_values, input_ids=None, attention_mask=None, 
                       texts=None, return_loss=False):
                batch_size = pixel_values.shape[0]
                return {
                    'loss_value': torch.tensor(0.5),
                    'img_embeds': torch.randn(batch_size, 512),
                    'text_embeds': torch.randn(batch_size, 512) if input_ids is not None else None,
                }
        
        dummy_model = DummyModel()
        trainer = VisionLanguageTrainer(
            model=dummy_model,
            config=test_config,
            output_dir='./test_checkpoints',
            experiment_name='test_trainer',
            use_wandb=False,
            seed=42
        )
        print("   ✅ Trainer initialized with dummy model")
    except Exception as e:
        print(f"   ❌ Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test helper methods
    print("\n5️⃣  Testing helper methods...")
    try:
        # Test _prepare_model_inputs
        batch = {
            'pixel_values': torch.randn(2, 3, 224, 224),
            'input_ids': torch.randint(0, 1000, (2, 50)),
            'attention_mask': torch.ones(2, 50),
        }
        model_inputs = trainer._prepare_model_inputs(batch)
        assert 'pixel_values' in model_inputs
        assert 'input_ids' in model_inputs
        assert 'return_loss' in model_inputs
        print("   ✅ _prepare_model_inputs works")
        
        # Test _extract_loss
        outputs = {'loss_value': torch.tensor(0.5)}
        loss = trainer._extract_loss(outputs)
        assert loss.item() == 0.5
        print("   ✅ _extract_loss works")
        
    except Exception as e:
        print(f"   ❌ Helper methods failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test convenience functions
    print("\n6️⃣  Testing convenience functions...")
    try:
        trainer_entrep = create_trainer_for_entrep(dummy_model, test_config, use_wandb=False)
        trainer_medclip = create_trainer_for_medclip(dummy_model, test_config, use_wandb=False)
        trainer_biomedclip = create_trainer_for_biomedclip(dummy_model, test_config, use_wandb=False)
        print("   ✅ All convenience functions work")
    except Exception as e:
        print(f"   ❌ Convenience functions failed: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    return True


if __name__ == '__main__':
    success = test_trainer_initialization()
    sys.exit(0 if success else 1)


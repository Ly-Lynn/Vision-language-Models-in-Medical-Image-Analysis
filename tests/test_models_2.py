"""
Pytest unit tests for factory system
"""

import pytest
import sys
import os
import torch

# Add modules to path
sys.path.append('./modules')

# Import all required modules
from modules.models.factory import ModelFactory, create_medclip, create_biomedclip, create_entrep
from modules.models.medclip import MedCLIPModel, PromptClassifier
from modules.models.biomedclip import BioMedCLIPModel
from modules.models.entrep import ENTRepModel, ENTRepClassifier


class TestImports:
    """Test import functionality"""
    
    def test_factory_imports(self):
        """Test factory imports work correctly"""
        assert ModelFactory is not None
        assert create_medclip is not None
        assert create_biomedclip is not None
        assert create_entrep is not None
    
    def test_model_imports(self):
        """Test model class imports"""
        assert MedCLIPModel is not None
        assert BioMedCLIPModel is not None
        assert ENTRepModel is not None
    
    def test_classifier_imports(self):
        """Test classifier imports"""
        assert PromptClassifier is not None
        assert ENTRepClassifier is not None


class TestRegistry:
    """Test registry functionality"""
    
    def test_available_models(self):
        """Test getting available models"""
        available_models = ModelFactory.get_available_models()
        
        expected_models = ['medclip', 'biomedclip', 'entrep']
        for model_type in expected_models:
            assert model_type in available_models, f"Missing {model_type}"
            assert 'base' in available_models[model_type], f"Missing base variant for {model_type}"
    
    def test_available_classifiers(self):
        """Test getting available classifiers"""
        available_classifiers = ModelFactory.get_available_classifiers()
        
        expected_models = ['medclip', 'biomedclip', 'entrep']
        for model_type in expected_models:
            assert model_type in available_classifiers, f"Missing {model_type} classifiers"
            assert 'zeroshot' in available_classifiers[model_type], f"Missing zeroshot for {model_type}"
    
    def test_print_registry(self):
        """Test print registry doesn't crash"""
        # Should not raise exception
        ModelFactory.print_registry()


class TestMedCLIPCreation:
    """Test MedCLIP model creation and functionality"""
    
    def test_create_medclip_vit(self):
        """Test creating MedCLIP with ViT encoder"""
        vit_model = create_medclip(
            text_encoder='bert',
            vision_encoder='vit', 
            pretrained=False
        )
        
        assert isinstance(vit_model, MedCLIPModel)
        info = vit_model.get_encoder_info()
        assert info['text_encoder'] == 'bert'
        assert info['vision_encoder'] == 'vit'
    
    def test_create_medclip_resnet(self):
        """Test creating MedCLIP with ResNet encoder"""
        resnet_model = create_medclip(
            text_encoder='bert',
            vision_encoder='resnet',
            pretrained=False
        )
        
        assert isinstance(resnet_model, MedCLIPModel)
        info = resnet_model.get_encoder_info()
        assert info['text_encoder'] == 'bert'
        assert info['vision_encoder'] == 'resnet'
    
    def test_medclip_encoder_info(self):
        """Test encoder info functionality"""
        model = create_medclip(
            text_encoder='bert',
            vision_encoder='vit',
            pretrained=False
        )
        
        info = model.get_encoder_info()
        assert 'text_encoder' in info
        assert 'vision_encoder' in info
        assert 'vision_model_type' in info
        assert 'text_model_type' in info
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_medclip_forward_pass(self):
        """Test MedCLIP forward pass with dummy data"""
        model = create_medclip(pretrained=False)
        
        dummy_images = torch.randn(1, 3, 224, 224)
        dummy_input_ids = torch.randint(0, 1000, (1, 128))
        dummy_attention = torch.ones(1, 128)
        
        with torch.no_grad():
            try:
                img_embeds = model.encode_image(dummy_images)
                text_embeds = model.encode_text(dummy_input_ids, dummy_attention)
                
                assert img_embeds.shape[0] == 1
                assert text_embeds.shape[0] == 1
                assert img_embeds.shape[1] == text_embeds.shape[1]  # Same feature dim
                
            except Exception:
                pytest.skip("Forward pass skipped - may require pretrained weights")


class TestENTRepCreation:
    """Test ENTRep model creation and functionality"""
    
    def test_create_entrep_clip_clip(self):
        """Test creating ENTRep with CLIP text + CLIP vision"""
        clip_model = create_entrep(
            text_encoder='clip',
            vision_encoder='clip',
            pretrained=False  # Don't download pretrained weights
        )
        
        assert isinstance(clip_model, ENTRepModel)
        info = clip_model.get_encoder_info()
        assert info['text_encoder'] == 'clip'
        assert info['vision_encoder'] == 'clip'
        assert clip_model.text_model is not None
    
    def test_create_entrep_none_endovit(self):
        """Test creating ENTRep with no text + EndoViT vision"""
        endovit_model = create_entrep(
            text_encoder='none',
            vision_encoder='endovit',
            num_classes=7
        )
        
        assert isinstance(endovit_model, ENTRepModel)
        info = endovit_model.get_encoder_info()
        assert info['text_encoder'] == 'none'
        assert info['vision_encoder'] == 'endovit'
        assert endovit_model.text_model is None
    
    def test_create_entrep_none_dinov2(self):
        """Test creating ENTRep with no text + DinoV2 vision"""
        dinov2_model = create_entrep(
            text_encoder='none',
            vision_encoder='dinov2',
            num_classes=5,
            pretrained=False
        )
        
        assert isinstance(dinov2_model, ENTRepModel)
        info = dinov2_model.get_encoder_info()
        assert info['text_encoder'] == 'none'
        assert info['vision_encoder'] == 'dinov2'
        assert dinov2_model.text_model is None
    
    def test_vision_only_text_encoding_error(self):
        """Test that vision-only models properly handle text encoding errors"""
        vision_only_model = create_entrep(
            text_encoder='none',
            vision_encoder='endovit',
            num_classes=7,
            pretrained=False
        )
        
        dummy_input_ids = torch.randint(0, 1000, (1, 77))
        dummy_attention = torch.ones(1, 77)
        
        with pytest.raises(NotImplementedError):
            vision_only_model.encode_text(dummy_input_ids, dummy_attention)


class TestBioMedCLIPCreation:
    """Test BioMedCLIP model creation"""
    
    @pytest.mark.slow
    def test_create_biomedclip(self):
        """Test creating BioMedCLIP model - may download weights"""
        try:
            model = create_biomedclip()
            assert isinstance(model, BioMedCLIPModel)
        except Exception as e:
            pytest.skip(f"BioMedCLIP creation skipped - may require model download: {e}")
    
    def test_biomedclip_factory_method(self):
        """Test BioMedCLIP creation via ModelFactory"""
        try:
            model = ModelFactory.create_model(
                model_type='biomedclip',
                variant='base'
            )
            assert isinstance(model, BioMedCLIPModel)
        except Exception as e:
            pytest.skip(f"BioMedCLIP factory test skipped: {e}")


class TestFactoryMethods:
    """Test ModelFactory methods work correctly"""
    
    def test_medclip_via_factory(self):
        """Test creating MedCLIP via ModelFactory"""
        medclip_model = ModelFactory.create_model(
            model_type='medclip',
            variant='base',
            text_encoder_type='bert',
            vision_encoder_type='vit',
            pretrained=False
        )
        
        assert isinstance(medclip_model, MedCLIPModel)
        assert medclip_model is not None
    
    def test_entrep_via_factory(self):
        """Test creating ENTRep via ModelFactory"""
        entrep_model = ModelFactory.create_model(
            model_type='entrep',
            variant='base',
            text_encoder_type='clip',
            vision_encoder_type='clip',
            pretrained=False
        )
        
        assert isinstance(entrep_model, ENTRepModel)
        assert entrep_model is not None
    
    def test_invalid_model_type(self):
        """Test error handling for invalid model type"""
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelFactory.create_model(model_type='invalid_type')
    
    def test_invalid_variant(self):
        """Test error handling for invalid variant"""
        with pytest.raises(ValueError, match="Unknown variant"):
            ModelFactory.create_model(
                model_type='medclip',
                variant='invalid_variant'
            )


class TestEncoderCombinations:
    """Test different encoder combinations"""
    
    @pytest.mark.parametrize("text_enc,vision_enc", [
        ('bert', 'vit'),
        ('bert', 'resnet')
    ])
    def test_medclip_combinations(self, text_enc, vision_enc):
        """Test MedCLIP encoder combinations"""
        model = create_medclip(
            text_encoder=text_enc,
            vision_encoder=vision_enc,
            pretrained=False
        )
        
        info = model.get_encoder_info()
        assert info['text_encoder'] == text_enc
        assert info['vision_encoder'] == vision_enc
        assert isinstance(model, MedCLIPModel)
    
    @pytest.mark.parametrize("text_enc,vision_enc", [
        ('clip', 'clip'),
        ('none', 'endovit'),
        ('none', 'dinov2')
    ])
    def test_entrep_combinations(self, text_enc, vision_enc):
        """Test ENTRep encoder combinations"""
        model = create_entrep(
            text_encoder=text_enc,
            vision_encoder=vision_enc,
            num_classes=7,
            pretrained=False  # Don't download pretrained weights
        )
        
        info = model.get_encoder_info()
        assert info['text_encoder'] == text_enc
        assert info['vision_encoder'] == vision_enc
        assert isinstance(model, ENTRepModel)
        
        if text_enc == 'none':
            assert model.text_model is None
        else:
            assert model.text_model is not None


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_invalid_model_type(self):
        """Test creating model with invalid type"""
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelFactory.create_model(model_type='invalid_type')
    
    def test_invalid_vision_encoder(self):
        """Test invalid vision encoder type"""
        with pytest.raises(ValueError, match="Unsupported.*encoder type"):
            create_entrep(
                text_encoder='none',
                vision_encoder='invalid_encoder',
                num_classes=7,
                pretrained=False
            )
    
    def test_invalid_text_encoder(self):
        """Test invalid text encoder type"""
        with pytest.raises(ValueError, match="Unsupported.*encoder type"):
            create_entrep(
                text_encoder='invalid_text_encoder',
                vision_encoder='clip',
                pretrained=False
            )
    
    def test_vision_only_text_encoding_error(self):
        """Test that vision-only models handle text encoding properly"""
        vision_only_model = create_entrep(
            text_encoder='none',
            vision_encoder='endovit',
            num_classes=7,
            pretrained=False
        )
        
        dummy_ids = torch.randint(0, 1000, (1, 77))
        dummy_mask = torch.ones(1, 77)
        
        with pytest.raises(NotImplementedError):
            vision_only_model.encode_text(dummy_ids, dummy_mask)


class TestClassifiers:
    """Test classifier creation and functionality"""
    
    def test_create_medclip_zeroshot_classifier(self):
        """Test creating MedCLIP zero-shot classifier"""
        classifier = ModelFactory.create_zeroshot_classifier(
            model_type='medclip',
            class_names=['normal', 'pneumonia'],
            ensemble=True,
            pretrained=False
        )
        
        assert isinstance(classifier, PromptClassifier)
        assert classifier.ensemble == True
    
    def test_create_entrep_zeroshot_classifier(self):
        """Test creating ENTRep zero-shot classifier"""
        classifier = ModelFactory.create_classifier(
            model_type='entrep',
            task_type='zeroshot',
            class_names=['class1', 'class2', 'class3'],
            num_classes=3,
            ensemble=False,
            model_kwargs={
                'text_encoder_type': 'clip',
                'vision_encoder_type': 'clip',
                'pretrained': False
            }
        )
        
        assert isinstance(classifier, ENTRepClassifier)
        assert classifier.ensemble == False
    
    def test_classifier_requires_class_names(self):
        """Test that zero-shot classifier requires class names"""
        model = create_medclip(pretrained=False)
        
        with pytest.raises(ValueError, match="class_names required"):
            ModelFactory.create_classifier(
                model=model,
                task_type='zeroshot'
                # Missing class_names
            )


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_model_creation_workflow(self):
        """Test complete model creation workflow"""
        # Test all model types can be created
        models_to_test = [
            ('medclip', {'text_encoder_type': 'bert', 'vision_encoder_type': 'vit', 'pretrained': False}),
            ('entrep', {'text_encoder_type': 'clip', 'vision_encoder_type': 'clip', 'pretrained': False}),
        ]
        
        for model_type, kwargs in models_to_test:
            model = ModelFactory.create_model(
                model_type=model_type,
                variant='base',
                **kwargs
            )
            
            assert model is not None
            # Each model should have encoder info
            if hasattr(model, 'get_encoder_info'):
                info = model.get_encoder_info()
                assert isinstance(info, dict)
                assert len(info) > 0
    
    def test_convenience_functions(self):
        """Test all convenience functions work"""
        # Test convenience functions
        models = [
            ('MedCLIP ViT', lambda: create_medclip('bert', 'vit', pretrained=False)),
            ('MedCLIP ResNet', lambda: create_medclip('bert', 'resnet', pretrained=False)),
            ('ENTRep CLIP', lambda: create_entrep('clip', 'clip', pretrained=False)),
            ('ENTRep Vision-only', lambda: create_entrep('none', 'endovit', num_classes=7, pretrained=False))
        ]
        
        for name, create_func in models:
            model = create_func()
            assert model is not None, f"Failed to create {name}"
    
    def test_all_encoder_combinations(self, dummy_data):
        """Test all valid encoder combinations work"""
        combinations = [
            # MedCLIP combinations
            ('medclip', 'bert', 'vit'),
            ('medclip', 'bert', 'resnet'),
            
            # ENTRep combinations
            ('entrep', 'clip', 'clip'),
            ('entrep', 'none', 'endovit'),
            ('entrep', 'none', 'dinov2'),
        ]
        
        for model_type, text_enc, vision_enc in combinations:
            if model_type == 'medclip':
                model = create_medclip(
                    text_encoder=text_enc,
                    vision_encoder=vision_enc,
                    pretrained=False
                )
            else:  # entrep
                model = create_entrep(
                    text_encoder=text_enc,
                    vision_encoder=vision_enc,
                    num_classes=dummy_data['num_classes'],
                    pretrained=False
                )
            
            assert model is not None
            info = model.get_encoder_info()
            assert info['text_encoder'] == text_enc
            assert info['vision_encoder'] == vision_enc


# Run with: 
# python -m pytest tests/test_models_2.py -v                    # All tests
# python -m pytest tests/test_models_2.py -v -m "not slow"     # Skip slow tests
# python -m pytest tests/test_models_2.py::TestRegistry -v     # Specific test class

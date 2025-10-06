import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factory import DatasetFactory
from utils import print_dataset_summary
from logging_config import get_logger

logger = get_logger(__name__)


def demo_basic_usage():
    """
    Demo cách sử dụng cơ bản
    """
    print("🚀 Basic Usage Demo")
    print("=" * 50)
    
    # 1. Tạo dataset trực tiếp
    print("\n📊 1. Creating Datasets:")
    
    try:
        # MIMIC dataset cho MedCLIP
        mimic_dataset = DatasetFactory.create_dataset(
            dataset_name='mimic',
            dataset_type='classification',
            model_type='medclip',
            split='test'
        )
        print(f"  ✅ MIMIC (MedCLIP): {len(mimic_dataset)} samples")
        print(f"     Classes: {mimic_dataset.get_class_names()}")
        
    except Exception as e:
        print(f"  ❌ MIMIC (MedCLIP): {e}")
        
    try:
        # COVID dataset cho BiomedCLIP
        covid_dataset = DatasetFactory.create_dataset(
            dataset_name='covid',
            dataset_type='classification',
            model_type='biomedclip',
            split='test'
        )
        print(f"  ✅ COVID (BiomedCLIP): {len(covid_dataset)} samples")
        print(f"     Classes: {covid_dataset.get_class_names()}")
        
    except Exception as e:
        print(f"  ❌ COVID (BiomedCLIP): {e}")
        
    # 2. Tạo DataLoader
    print("\n🔧 2. Creating DataLoaders:")
    
    try:
        # Zero-shot classification với MedCLIP
        dataloader = DatasetFactory.create_dataloader(
            dataset_name='mimic',
            task_type='zeroshot',
            model_type='medclip',
            batch_size=4,
            shuffle=False
        )
        
        # Test một batch
        for batch in dataloader:
            print(f"  ✅ MIMIC Zero-shot (MedCLIP):")
            print(f"     Batch size: {batch['pixel_values'].shape[0]}")
            print(f"     Image shape: {batch['pixel_values'].shape}")
            print(f"     Labels shape: {batch['labels'].shape}")
            print(f"     Classes: {len(batch['class_names'])}")
            break
            
    except Exception as e:
        print(f"  ❌ MIMIC Zero-shot (MedCLIP): {e}")


def demo_convenience_functions():
    """
    Demo các convenience functions
    """
    print("\n🎯 Convenience Functions Demo")
    print("=" * 50)
    
    from datasets import (
        create_mimic_dataloader,
        create_covid_dataloader,
        create_rsna_dataloader,
        create_contrastive_dataloader
    )
    
    convenience_functions = [
        ('MIMIC Zero-shot', lambda: create_mimic_dataloader(
            task_type='zeroshot',
            model_type='medclip',
            batch_size=2
        )),
        ('COVID Supervised', lambda: create_covid_dataloader(
            task_type='supervised',
            model_type='biomedclip',
            batch_size=2
        )),
        ('RSNA Zero-shot', lambda: create_rsna_dataloader(
            task_type='zeroshot',
            model_type='medclip',
            batch_size=2
        )),
        ('MIMIC Contrastive', lambda: create_contrastive_dataloader(
            dataset_name='mimic',
            model_type='medclip',
            split='train',
            batch_size=2
        ))
    ]
    
    for name, func in convenience_functions:
        try:
            dataloader = func()
            for batch in dataloader:
                print(f"  ✅ {name}: batch shape {batch['pixel_values'].shape}")
                break
        except Exception as e:
            print(f"  ❌ {name}: {e}")


def demo_model_comparison():
    """
    Demo so sánh giữa MedCLIP và BiomedCLIP
    """
    print("\n🔬 Model Comparison Demo")
    print("=" * 50)
    
    datasets_to_compare = ['mimic', 'covid', 'rsna']
    
    for dataset_name in datasets_to_compare:
        print(f"\n📊 {dataset_name.upper()} Dataset:")
        
        for model_type in ['medclip', 'biomedclip']:
            try:
                dataloader = DatasetFactory.create_dataloader(
                    dataset_name=dataset_name,
                    task_type='zeroshot',
                    model_type=model_type,
                    batch_size=2
                )
                
                for batch in dataloader:
                    print(f"  {model_type.upper()}: {batch['pixel_values'].shape} -> {batch['labels'].shape}")
                    break
                    
            except Exception as e:
                print(f"  {model_type.upper()}: Error - {e}")


def demo_advanced_usage():
    """
    Demo advanced usage với custom parameters
    """
    print("\n🎛️ Advanced Usage Demo")
    print("=" * 50)
    
    # Custom class prompts
    custom_covid_prompts = {
        'COVID': [
            'covid-19 pneumonia',
            'coronavirus infection',
            'sars-cov-2 findings',
            'covid pneumonia',
            'viral pneumonia covid'
        ],
        'Normal': [
            'normal chest xray',
            'clear lungs',
            'no abnormalities',
            'healthy chest',
            'normal findings'
        ]
    }
    
    try:
        # Tạo COVID dataloader với custom prompts
        dataloader = DatasetFactory.create_dataloader(
            dataset_name='covid',
            task_type='zeroshot',
            model_type='medclip',
            batch_size=2,
            cls_prompts=custom_covid_prompts,
            template='this is a chest x-ray showing ',
            n_prompt=3
        )
        
        for batch in dataloader:
            print(f"  ✅ Custom COVID prompts: {batch['pixel_values'].shape}")
            print(f"     Prompt classes: {batch['class_names']}")
            break
            
    except Exception as e:
        print(f"  ❌ Custom COVID prompts: {e}")
        
    # Custom transforms
    try:
        from torchvision import transforms
        
        custom_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Larger size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
        
        dataset = DatasetFactory.create_dataset(
            dataset_name='rsna',
            model_type='medclip',
            transform=custom_transform
        )
        
        print(f"  ✅ Custom transforms: {len(dataset)} samples")
        
        # Test image shape
        if len(dataset) > 0:
            img, labels = dataset[0]
            print(f"     Custom image shape: {img.shape}")
            
    except Exception as e:
        print(f"  ❌ Custom transforms: {e}")


def main():
    """
    Main demo function
    """
    print("🏥 Medical Datasets Demo")
    print("=" * 70)
    
    # Print dataset summary
    print_dataset_summary()
    
    # Print factory registry
    print("\n🏭 Factory Registry:")
    DatasetFactory.print_registry()
    
    # Run demos
    demo_basic_usage()
    demo_convenience_functions()
    demo_model_comparison()
    demo_advanced_usage()
    
    print("\n" + "=" * 70)
    print("✅ Demo completed successfully!")
    print("\n📖 Usage Examples:")
    print("   # Basic usage")
    print("   from src.datasets import DatasetFactory")
    print("   dataloader = DatasetFactory.create_dataloader('mimic', 'zeroshot', 'medclip')")
    print("")
    print("   # Convenience functions")
    print("   from src.datasets import create_covid_dataloader")
    print("   dataloader = create_covid_dataloader(task_type='supervised', model_type='biomedclip')")
    print("")
    print("   # Custom parameters")
    print("   dataloader = DatasetFactory.create_dataloader(")
    print("       dataset_name='rsna',")
    print("       task_type='zeroshot',")
    print("       model_type='medclip',")
    print("       batch_size=32,")
    print("       cls_prompts=custom_prompts,")
    print("       template='custom template'")
    print("   )")


if __name__ == "__main__":
    main()

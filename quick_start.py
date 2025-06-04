"""
SGWCN Quick Start Script
Test data loading and run a quick training example
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import create_dataloaders, ModelNet40Dataset
from src.models import SGWCNClassifier
from configs.sgwcn_config import ConfigPresets


def test_data_loading():
    """Test ModelNet40 data loading"""
    print("=" * 50)
    print("Testing Data Loading")
    print("=" * 50)
    
    data_root = "data/modelnet40_ply_hdf5_2048"
    
    if not os.path.exists(data_root):
        print(f"❌ Data not found at {data_root}")
        print("Please ensure ModelNet40 data is placed in the data/ directory")
        return False
    
    try:
        # Test dataset creation
        dataset = ModelNet40Dataset(
            data_root=data_root,
            split='train',
            num_points=512,  # Use fewer points for quick test
            normalize=True,
            augmentation=True
        )
        
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        print(f"✓ Number of classes: {dataset.num_classes}")
        print(f"✓ Class names: {dataset.class_names[:5]}...")
        
        # Test a sample
        points, label = dataset[0]
        print(f"✓ Sample shape: {points.shape}")
        print(f"✓ Sample label: {label} ({dataset.class_names[label]})")
        print(f"✓ Point range: [{points.min():.3f}, {points.max():.3f}]")
        
        # Test data loaders with small batch
        train_loader, test_loader = create_dataloaders(
            data_root=data_root,
            batch_size=4,
            num_points=512,
            num_workers=0,  # No multiprocessing for testing
            cache_data=False
        )
        
        print(f"✓ Train batches: {len(train_loader)}")
        print(f"✓ Test batches: {len(test_loader)}")
        
        # Test a batch
        batch_points, batch_labels = next(iter(train_loader))
        print(f"✓ Batch points shape: {batch_points.shape}")
        print(f"✓ Batch labels shape: {batch_labels.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False


def test_model_creation():
    """Test SGWCN model creation and forward pass"""
    print("\n" + "=" * 50)
    print("Testing Model Creation")
    print("=" * 50)
    
    try:
        # Create model with debug configuration
        model = SGWCNClassifier(
            input_dim=3,
            hidden_dims=[32, 64],  # Small for quick test
            num_classes=40,
            num_time_steps=4,      # Fewer time steps
            k_neighbors=10,        # Fewer neighbors
            num_points=512
        )
        
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        batch_size = 2
        num_points = 512
        point_cloud = torch.randn(batch_size, num_points, 3)
        
        print(f"✓ Input shape: {point_cloud.shape}")
        
        with torch.no_grad():
            logits = model(point_cloud)
            print(f"✓ Output shape: {logits.shape}")
            print(f"✓ Output range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        
        # Test with analysis
        analysis = model.forward_with_analysis(point_cloud)
        print(f"✓ Analysis keys: {list(analysis.keys())}")
        print(f"✓ Number of layers analyzed: {len(analysis['layer_outputs'])}")
        print(f"✓ Energy consumption: {analysis['energy_consumption']:.2e} J")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False


def quick_training_demo():
    """Run a very quick training demo"""
    print("\n" + "=" * 50)
    print("Quick Training Demo")
    print("=" * 50)
    
    data_root = "data/modelnet40_ply_hdf5_2048"
    
    if not os.path.exists(data_root):
        print("❌ Skipping training demo - data not found")
        return False
    
    try:
        # Use debug configuration
        config = ConfigPresets.fast_debug()
        config.num_epochs = 2  # Very quick demo
        config.batch_size = 4
        config.print_freq = 5
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"✓ Using device: {config.device}")
        
        # Create data loaders
        train_loader, test_loader = create_dataloaders(
            data_root=data_root,
            batch_size=config.batch_size,
            num_points=config.num_points,
            num_workers=0,
            cache_data=False
        )
        
        # Create model
        model = SGWCNClassifier(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            num_classes=config.num_classes,
            num_time_steps=config.num_time_steps,
            k_neighbors=config.k_neighbors,
            num_points=config.num_points
        )
        
        model = model.to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training loop
        model.train()
        total_loss = 0.0
        num_batches = min(10, len(train_loader))  # Limit to 10 batches
        
        print(f"✓ Training for {num_batches} batches...")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
                
            data, target = data.to(config.device), target.to(config.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"✓ Average training loss: {avg_loss:.4f}")
        
        # Quick validation
        model.eval()
        correct = 0
        total = 0
        val_batches = min(5, len(test_loader))
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if batch_idx >= val_batches:
                    break
                    
                data, target = data.to(config.device), target.to(config.device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100.0 * correct / total
        print(f"✓ Quick validation accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        return True
        
    except Exception as e:
        print(f"❌ Training demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run quick start tests"""
    print("🚀 SGWCN Quick Start")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🎯 GPU: {torch.cuda.get_device_name()}")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_data_loading():
        tests_passed += 1
    
    if test_model_creation():
        tests_passed += 1
        
    if quick_training_demo():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"✅ Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! SGWCN is ready for training.")
        print("\nNext steps:")
        print("1. Run full training: python train_sgwcn.py --config fast_debug")
        print("2. Or customize training: python train_sgwcn.py --batch_size 16 --num_epochs 50")
        print("3. Monitor with tensorboard: tensorboard --logdir logs")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("Make sure:")
        print("1. ModelNet40 data is in data/modelnet40_ply_hdf5_2048/")
        print("2. All dependencies are installed: pip install -r requirements.txt")
        print("3. PyTorch and CUDA are properly configured")


if __name__ == "__main__":
    main() 
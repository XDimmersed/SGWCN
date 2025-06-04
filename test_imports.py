#!/usr/bin/env python3
"""
Simple import test to verify all modules can be imported correctly
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all critical imports"""
    
    print("🧪 Testing imports...")
    
    try:
        # Test core model imports
        from src.models import SGWCNClassifier, SpikingGraphWaveletNet
        print("✓ Model imports successful")
        
        # Test data imports
        from src.data.dataset import ModelNet40Dataset, create_dataloaders
        print("✓ Data imports successful")
        
        # Test utils imports
        from src.utils import knn_search, compute_local_density, poisson_encoding
        print("✓ Utils imports successful")
        
        # Test config imports
        from configs.sgwcn_config import SGWCNConfig, ConfigPresets
        print("✓ Config imports successful")
        
        # Test trainer imports
        from src.train.trainer import SGWCNTrainer
        from src.train.metrics import compute_metrics
        print("✓ Training imports successful")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without requiring data"""
    
    print("\n🔧 Testing basic functionality...")
    
    try:
        import torch
        from src.models import SGWCNClassifier
        from configs.sgwcn_config import ConfigPresets
        
        # Create a simple model
        config = ConfigPresets.fast_debug()
        model = SGWCNClassifier(
            num_classes=40,
            num_points=256,
            hidden_dims=[32, 64],
            num_time_steps=4
        )
        
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass with dummy data
        batch_size = 2
        num_points = 256
        dummy_input = torch.randn(batch_size, num_points, 3)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✓ Forward pass successful: {dummy_input.shape} -> {output.shape}")
        
        print("\n🎉 Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🚀 SGWCN Import and Functionality Test")
    print("=" * 50)
    
    # Test imports
    import_success = test_imports()
    
    if import_success:
        # Test basic functionality
        func_success = test_basic_functionality()
    else:
        func_success = False
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  Imports: {'✅ PASS' if import_success else '❌ FAIL'}")
    print(f"  Functionality: {'✅ PASS' if func_success else '❌ FAIL'}")
    
    if import_success and func_success:
        print("\n🎉 All tests passed! SGWCN is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python quick_start.py")
        print("  2. Or: python run_sgwcn.py test")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit(main()) 
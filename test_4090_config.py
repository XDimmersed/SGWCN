#!/usr/bin/env python3
"""
Test RTX 4090 Optimized Configuration
Quick validation of memory usage and model functionality
"""

import torch
import torch.nn as nn
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import SGWCNClassifier
from configs.sgwcn_config import ConfigPresets


def test_4090_config():
    """Test RTX 4090 optimized configuration"""
    print("🚀 Testing RTX 4090 Optimized Configuration")
    print("=" * 50)
    
    # Get configuration
    config = ConfigPresets.rtx4090_optimized()
    
    print("📋 Configuration Summary:")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Hidden dims: {config.hidden_dims}")
    print(f"  - Num points: {config.num_points}")
    print(f"  - Time steps: {config.num_time_steps}")
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - Use AMP: {config.use_amp}")
    print(f"  - Cache data: {config.cache_data}")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
        
    device = torch.device('cuda')
    print(f"🔧 Using device: {device}")
    print(f"🔧 GPU: {torch.cuda.get_device_name()}")
    print(f"🔧 Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1e9
    print(f"🔧 Initial GPU memory: {initial_memory:.2f} GB")
    
    try:
        # Create model
        print("\n🔨 Creating SGWCN model...")
        model = SGWCNClassifier(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            num_classes=config.num_classes,
            num_time_steps=config.num_time_steps,
            k_neighbors=config.k_neighbors,
            chebyshev_order=config.chebyshev_order,
            beta=config.beta,
            lambda_param=config.lambda_param,
            epsilon=config.epsilon,
            tau_mem=config.tau_mem,
            theta_pos=config.theta_pos,
            theta_neg=config.theta_neg,
            dropout=config.dropout,
            use_faiss=config.use_faiss,
            readout_mode=config.readout_mode
        ).to(device)
        
        model_memory = torch.cuda.memory_allocated() / 1e9
        print(f"✅ Model created successfully!")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"📊 Model memory: {model_memory - initial_memory:.2f} GB")
        
        # Test forward pass
        print("\n🧪 Testing forward pass...")
        batch_size = config.batch_size
        num_points = config.num_points
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, num_points, 3, device=device)
        
        # Forward pass
        start_time = time.time()
        with torch.no_grad():
            output = model(dummy_input)
        forward_time = time.time() - start_time
        
        max_memory = torch.cuda.max_memory_allocated() / 1e9
        current_memory = torch.cuda.memory_allocated() / 1e9
        
        print(f"✅ Forward pass successful!")
        print(f"📊 Input shape: {dummy_input.shape}")
        print(f"📊 Output shape: {output.shape}")
        print(f"📊 Forward time: {forward_time:.3f}s")
        print(f"📊 Peak GPU memory: {max_memory:.2f} GB")
        print(f"📊 Current GPU memory: {current_memory:.2f} GB")
        print(f"📊 Memory utilization: {max_memory/24:.1%}")
        
        # Test backward pass
        print("\n🧪 Testing backward pass...")
        model.train()
        
        # Create dummy target
        dummy_target = torch.randint(0, config.num_classes, (batch_size,), device=device)
        
        # Forward + backward
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        output = model(dummy_input)
        loss = nn.CrossEntropyLoss()(output, dummy_target)
        loss.backward()
        
        backward_time = time.time() - start_time
        peak_memory_backward = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"✅ Backward pass successful!")
        print(f"📊 Loss value: {loss.item():.4f}")
        print(f"📊 Backward time: {backward_time:.3f}s")
        print(f"📊 Peak memory (training): {peak_memory_backward:.2f} GB")
        print(f"📊 Training memory utilization: {peak_memory_backward/24:.1%}")
        
        # Memory efficiency analysis
        print(f"\n📈 Memory Efficiency Analysis:")
        print(f"  - Model size: {(model_memory - initial_memory):.2f} GB")
        print(f"  - Forward pass overhead: {(max_memory - model_memory):.2f} GB") 
        print(f"  - Training overhead: {(peak_memory_backward - max_memory):.2f} GB")
        print(f"  - Total training memory: {peak_memory_backward:.2f} GB / 24 GB")
        
        if peak_memory_backward < 20:  # Leave 4GB headroom
            print("✅ Configuration is memory-safe for training!")
        else:
            print("⚠️  Configuration may be tight on memory")
            
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print(f"📊 GPU memory at error: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        return False
    
    finally:
        # Cleanup
        torch.cuda.empty_cache()


if __name__ == "__main__":
    success = test_4090_config()
    if success:
        print("\n🎉 RTX 4090 configuration test passed!")
        print("💡 You can now run: python train_sgwcn.py --config rtx4090_optimized")
    else:
        print("\n❌ Configuration test failed!")
        print("💡 Consider using: python train_sgwcn.py --config fast_debug") 
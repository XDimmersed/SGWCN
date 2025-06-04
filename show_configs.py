#!/usr/bin/env python3
"""
Display and compare all available SGWCN configuration presets
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.sgwcn_config import ConfigPresets, get_debug_stable_config


def display_config_comparison():
    """Display comparison of all configuration presets"""
    print("🔧 SGWCN Configuration Presets Comparison")
    print("=" * 80)
    
    # Get all configurations
    configs = {
        'default': ConfigPresets.default(),
        'fast_debug': ConfigPresets.fast_debug(), 
        'debug_stable': get_debug_stable_config(),
        'rtx4090_optimized': ConfigPresets.rtx4090_optimized(),
        'rtx4090_safe': ConfigPresets.rtx4090_safe(),
        'high_performance': ConfigPresets.high_performance(),
        'energy_efficient': ConfigPresets.energy_efficient()
    }
    
    # Key parameters to compare
    params = [
        'batch_size',
        'hidden_dims', 
        'num_epochs',
        'num_time_steps',
        'num_points',
        'k_neighbors',
        'chebyshev_order',
        'learning_rate',
        'use_amp',
        'cache_data',
        'tau_mem',
        'theta_pos',
        'dropout'
    ]
    
    # Display header
    print(f"{'Parameter':<20}", end='')
    for name in configs.keys():
        print(f"{name:<16}", end='')
    print()
    print("-" * 140)
    
    # Display each parameter
    for param in params:
        print(f"{param:<20}", end='')
        for config in configs.values():
            value = getattr(config, param)
            if isinstance(value, list):
                value_str = str(value)[:12] + '..' if len(str(value)) > 14 else str(value)
            else:
                value_str = str(value)[:14]
            print(f"{value_str:<16}", end='')
        print()
    
    print("\n" + "=" * 80)
    print("📊 Memory Estimation (approximate)")
    print("=" * 80)
    
    # Estimate memory usage for each config
    for name, config in configs.items():
        B = config.batch_size
        N = config.num_points  
        K = config.chebyshev_order + 1
        F_max = max(config.hidden_dims)
        
        # Theta_local tensor: [B, N, K, F_in, F_out]
        memory_gb = (B * N * K * F_max * F_max * 4) / (1024**3)
        
        # Rough total estimate
        total_est = memory_gb * 2.5
        
        status = "✅ Safe" if total_est < 20 else "⚠️ Tight" if total_est < 24 else "❌ Risky"
        
        print(f"{name:<20} | Peak: ~{total_est:.1f} GB | {status}")
    
    print("\n" + "=" * 80)
    print("💡 Recommendations")
    print("=" * 80)
    print("🚀 Quick testing:      fast_debug (5 epochs)")
    print("🔧 Stable debugging:   debug_stable (memory safe)")
    print("🎯 RTX 4090 balanced:  rtx4090_optimized (performance)")  
    print("🛡️ RTX 4090 safe:     rtx4090_safe (memory conservative)")
    print("🏆 High performance:   high_performance (if you have huge GPU)")
    print("⚡ Energy efficient:   energy_efficient (fewer time steps)")
    print("📄 Default:           default (original paper settings)")
    
    print("\n📝 Usage examples:")
    print("python train_sgwcn.py --config fast_debug")
    print("python train_sgwcn.py --config rtx4090_safe")
    print("python train_sgwcn.py --config rtx4090_optimized")


if __name__ == "__main__":
    display_config_comparison() 
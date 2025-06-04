#!/usr/bin/env python3
"""
SGWCN Main Control Script
Unified entry point for training, evaluation, and testing
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_quick_test():
    """Run quick functionality test"""
    print("🚀 Running quick functionality test...")
    return subprocess.run([sys.executable, "quick_start.py"], cwd=os.getcwd()).returncode

def run_training(args):
    """Run training with specified arguments"""
    print("🎯 Starting SGWCN training...")
    
    cmd = [sys.executable, "train_sgwcn.py"]
    
    # Add arguments
    if args.config:
        cmd.extend(["--config", args.config])
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.learning_rate:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    if args.num_epochs:
        cmd.extend(["--num_epochs", str(args.num_epochs)])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    return subprocess.run(cmd, cwd=os.getcwd()).returncode

def run_evaluation(args):
    """Run model evaluation"""
    print("📊 Starting model evaluation...")
    
    if not args.model_path:
        print("❌ Error: Model path is required for evaluation")
        return 1
    
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found: {args.model_path}")
        return 1
    
    cmd = [sys.executable, "evaluate_sgwcn.py", args.model_path]
    
    if args.config_file:
        cmd.extend(["--config", args.config_file])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.save_dir:
        cmd.extend(["--save_dir", args.save_dir])
    if args.quick:
        cmd.append("--quick")
    
    return subprocess.run(cmd, cwd=os.getcwd()).returncode

def check_environment():
    """Check if environment is properly set up"""
    print("🔍 Checking environment...")
    
    issues = []
    
    # Check Python packages
    required_packages = ['torch', 'torch_geometric', 'h5py', 'numpy', 'matplotlib']
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            issues.append(f"Missing package: {package}")
            print(f"  ❌ {package}")
    
    # Check data directory
    data_dir = Path("data/modelnet40_ply_hdf5_2048")
    if data_dir.exists():
        print(f"  ✓ Data directory: {data_dir}")
        
        # Check key files
        required_files = ["train_files.txt", "test_files.txt", "shape_names.txt"]
        for file in required_files:
            file_path = data_dir / file
            if file_path.exists():
                print(f"    ✓ {file}")
            else:
                issues.append(f"Missing data file: {file}")
                print(f"    ❌ {file}")
    else:
        issues.append(f"Data directory not found: {data_dir}")
        print(f"  ❌ Data directory: {data_dir}")
    
    # Check directories
    required_dirs = ["src", "configs"]
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"  ✓ {dir_name}/")
        else:
            issues.append(f"Missing directory: {dir_name}")
            print(f"  ❌ {dir_name}/")
    
    if issues:
        print("\n❌ Environment check failed:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nTo fix:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Ensure ModelNet40 data is in data/modelnet40_ply_hdf5_2048/")
        return False
    else:
        print("\n✅ Environment check passed!")
        return True

def show_help():
    """Show helpful usage information"""
    print("""
🚀 SGWCN - Spiking Graph Wavelet Convolution Network

Usage Examples:

  # Check environment
  python run_sgwcn.py check

  # Quick test
  python run_sgwcn.py test

  # Training
  python run_sgwcn.py train                                    # Default config
  python run_sgwcn.py train --config fast_debug                # Debug config
  python run_sgwcn.py train --batch_size 16 --num_epochs 100   # Custom params
  python run_sgwcn.py train --resume checkpoints/latest.pth    # Resume training

  # Evaluation
  python run_sgwcn.py eval checkpoints/best_model.pth          # Full evaluation
  python run_sgwcn.py eval checkpoints/best_model.pth --quick  # Quick evaluation

Configuration Presets:
  - default: Balanced performance and speed
  - fast_debug: Quick testing with small model
  - high_performance: Best accuracy (slower)
  - energy_efficient: Optimized for low power

Directories:
  - checkpoints/: Saved models
  - logs/: Training logs and tensorboard
  - evaluation_results/: Evaluation outputs
  - data/: ModelNet40 dataset
""")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="SGWCN Control Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Check command
    subparsers.add_parser('check', help='Check environment setup')
    
    # Test command
    subparsers.add_parser('test', help='Run quick functionality test')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train SGWCN model')
    train_parser.add_argument('--config', type=str, help='Configuration preset')
    train_parser.add_argument('--batch_size', type=int, help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, help='Learning rate')
    train_parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained model')
    eval_parser.add_argument('model_path', type=str, help='Path to model checkpoint')
    eval_parser.add_argument('--config_file', type=str, help='Config file path')
    eval_parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
    eval_parser.add_argument('--save_dir', type=str, help='Results save directory')
    eval_parser.add_argument('--quick', action='store_true', help='Quick evaluation')
    
    # Help command
    subparsers.add_parser('help', help='Show detailed help')
    
    args = parser.parse_args()
    
    if not args.command:
        show_help()
        return 0
    
    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("evaluation_results", exist_ok=True)
    
    if args.command == 'check':
        success = check_environment()
        return 0 if success else 1
    
    elif args.command == 'test':
        return run_quick_test()
    
    elif args.command == 'train':
        return run_training(args)
    
    elif args.command == 'eval':
        return run_evaluation(args)
    
    elif args.command == 'help':
        show_help()
        return 0
    
    else:
        print(f"❌ Unknown command: {args.command}")
        show_help()
        return 1

if __name__ == "__main__":
    exit(main()) 
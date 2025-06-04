"""
Configuration file for SGWCN training
Contains all hyperparameters and training settings
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SGWCNConfig:
    """Configuration for SGWCN training"""
    
    # Model architecture
    input_dim: int = 3
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    num_classes: int = 40
    num_time_steps: int = 10
    k_neighbors: int = 20
    chebyshev_order: int = 3
    
    # Graph construction parameters
    beta: float = 1.0
    lambda_param: float = 1.0
    epsilon: float = 1e-6
    use_faiss: bool = True
    
    # Spiking neuron parameters
    tau_mem: float = 20.0
    theta_pos: float = 1.0
    theta_neg: float = -1.0
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_epochs: int = 200
    warmup_epochs: int = 10
    
    # Data parameters
    data_root: str = "data/modelnet40_ply_hdf5_2048"
    num_points: int = 1024
    normalize: bool = True
    augmentation: bool = True
    cache_data: bool = False
    num_workers: int = 4
    
    # Optimization
    optimizer: str = "adam"  # "adam", "adamw", "sgd"
    scheduler: str = "cosine"  # "cosine", "step", "none"
    step_size: int = 50
    gamma: float = 0.1
    gradient_clip: float = 1.0
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1
    use_class_weights: bool = True
    
    # Checkpointing and logging
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_freq: int = 10
    eval_freq: int = 5
    print_freq: int = 50
    
    # Resume training
    resume: Optional[str] = None
    pretrained: Optional[str] = None
    
    # Device settings
    device: str = "cuda"
    seed: int = 42
    
    # Mixed precision and efficiency
    use_amp: bool = True
    compile_model: bool = False  # PyTorch 2.0 model compilation
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4
    
    # SNN specific
    surrogate_grad_beta: float = 1.0
    readout_mode: str = "rate"  # "rate", "count", "last"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Create directories
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Validate parameters
        assert self.num_classes > 0, "num_classes must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.theta_pos > 0, "theta_pos must be positive"
        assert self.theta_neg < 0, "theta_neg must be negative"
        
        # Set device
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            self.device = "cpu"
            self.use_amp = False
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    def save(self, filepath: str):
        """Save configuration to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from file"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# Predefined configurations for different scenarios
class ConfigPresets:
    """Predefined configuration presets"""
    
    @staticmethod
    def default():
        """Default configuration for ModelNet40"""
        return SGWCNConfig()
    
    @staticmethod
    def fast_debug():
        """Fast configuration for debugging"""
        return SGWCNConfig(
            hidden_dims=[32, 64],
            batch_size=8,
            num_epochs=5,
            num_time_steps=4,
            k_neighbors=10,
            num_points=512,
            save_freq=1,
            eval_freq=1,
            print_freq=10,
            num_workers=0
        )
    
    @staticmethod
    def high_performance():
        """High performance configuration"""
        return SGWCNConfig(
            hidden_dims=[128, 256, 512],
            batch_size=16,
            num_epochs=300,
            num_time_steps=12,
            learning_rate=0.0005,
            k_neighbors=32,
            chebyshev_order=4,
            tau_mem=30.0,
            cache_data=True
        )
    
    @staticmethod
    def energy_efficient():
        """Energy efficient configuration with lower time steps"""
        return SGWCNConfig(
            hidden_dims=[64, 128],
            num_time_steps=6,
            theta_pos=1.5,
            theta_neg=-1.5,
            tau_mem=15.0,
            batch_size=64
        )
    
    @staticmethod
    def small_dataset():
        """Configuration for smaller datasets"""
        return SGWCNConfig(
            hidden_dims=[32, 64, 128],
            batch_size=64,
            dropout=0.2,
            label_smoothing=0.0,
            weight_decay=1e-3,
            augmentation=True
        )
    
    @staticmethod
    def rtx4090_optimized():
        """Optimized configuration for RTX 4090 GPU - balanced performance"""
        return SGWCNConfig(
            # Model architecture - conservative for memory safety
            hidden_dims=[64, 96, 96],   # 进一步减小维度
            num_time_steps=6,           # 减少时间步数
            k_neighbors=16,             # 减少邻居数
            chebyshev_order=2,          # 降低Chebyshev阶数
            
            # Training parameters - 4090优化
            batch_size=8,               # 保守的批处理大小
            learning_rate=0.001,
            weight_decay=1e-4,
            num_epochs=150,             # 增加训练轮数以补偿小批次
            warmup_epochs=10,
            
            # Data parameters - 充分利用GPU
            num_points=1024,            # 保持完整点数
            normalize=True,
            augmentation=True,
            cache_data=True,            # 利用大显存缓存数据
            num_workers=8,              # 多进程加载
            
            # Optimization - 4090性能优化
            optimizer='adamw',          # 更好的优化器
            scheduler='cosine',
            gradient_clip=0.5,          # 更严格的梯度裁剪
            use_amp=True,              # 混合精度训练节省内存
            compile_model=False,        # 避免编译问题
            
            # SNN parameters - 平衡设置
            tau_mem=15.0,              # 更快的膜时间常数
            theta_pos=0.8,             # 稍微降低阈值
            theta_neg=-0.8,
            surrogate_grad_beta=1.0,
            
            # Regularization
            dropout=0.2,               # 增加dropout
            label_smoothing=0.05,      # 减少标签平滑
            use_class_weights=True,
            
            # Training efficiency
            save_freq=15,
            eval_freq=5,
            print_freq=20,
            early_stopping=True,
            patience=20,
            
            # Use FAISS for acceleration
            use_faiss=True
        )

    @staticmethod
    def rtx4090_safe():
        """Memory-safe configuration for RTX 4090 GPU - conservative but stable"""
        return SGWCNConfig(
            # Model architecture - very conservative for memory safety
            hidden_dims=[48, 80, 80],   # 更小的维度
            num_time_steps=6,           # 适中的时间步数
            k_neighbors=16,             # 适中的邻居数
            chebyshev_order=2,          # 较低的Chebyshev阶数
            
            # Training parameters - 安全优先
            batch_size=6,               # 更小的批处理大小
            learning_rate=0.0015,       # 稍高的学习率补偿小批次
            weight_decay=1e-4,
            num_epochs=200,             # 更多训练轮数
            warmup_epochs=15,
            
            # Data parameters - 平衡设置
            num_points=1024,            # 保持完整点数
            normalize=True,
            augmentation=True,
            cache_data=True,            # 利用大显存缓存数据
            num_workers=6,              # 适中的进程数
            
            # Optimization - 稳定性优先
            optimizer='adamw',
            scheduler='cosine',
            gradient_clip=0.3,          # 更严格的梯度裁剪
            use_amp=True,              # 混合精度训练
            compile_model=False,
            
            # SNN parameters - 保守设置
            tau_mem=12.0,              # 更快的膜时间常数
            theta_pos=0.6,             # 更低的阈值
            theta_neg=-0.6,
            surrogate_grad_beta=0.8,
            
            # Regularization
            dropout=0.25,              # 更多dropout
            label_smoothing=0.02,      # 很少的标签平滑
            use_class_weights=True,
            
            # Training efficiency
            save_freq=20,
            eval_freq=5,
            print_freq=15,
            early_stopping=True,
            patience=25,
            
            # Use FAISS for acceleration
            use_faiss=True
        )


def get_debug_stable_config():
    """Stable configuration for debugging training issues"""
    config = SGWCNConfig()
    
    # Model architecture - smaller for stability
    config.hidden_dims = [32, 64]
    config.num_time_steps = 16  # 增加时间步数 - 关键改进!
    config.k_neighbors = 10
    config.chebyshev_order = 2  # Reduce complexity
    
    # SNN parameters - 基于调试结果优化
    config.tau_mem = 10.0  # Faster membrane time constant
    config.theta_pos = 0.3  # 降低阈值 - 关键改进!
    config.theta_neg = -0.3  # 降低阈值 - 关键改进!
    config.surrogate_grad_beta = 0.5  # Sharper surrogate gradient
    
    # Training parameters - more conservative
    config.batch_size = 4  # Smaller batch size
    config.learning_rate = 0.0005  # 稍微提高学习率
    config.weight_decay = 0.0001
    config.gradient_clip = 0.5  # Stricter gradient clipping
    config.num_epochs = 20  # 更多训练轮数
    
    # Data loading
    config.num_workers = 2  # Enable parallel loading
    config.num_points = 256  # Fewer points for faster processing
    
    # Disable problematic features
    config.use_faiss = False  # Avoid FAISS issues
    config.use_amp = False  # Disable mixed precision for stability
    config.label_smoothing = 0.0  # Disable label smoothing
    config.use_class_weights = False  # Disable class weighting
    
    # More frequent logging
    config.print_freq = 5
    config.eval_freq = 1
    
    return config


if __name__ == "__main__":
    # Test configuration
    import torch
    
    config = SGWCNConfig()
    print("Default configuration:")
    print(config)
    
    # Test presets
    debug_config = ConfigPresets.fast_debug()
    print(f"\nDebug config batch size: {debug_config.batch_size}")
    
    # Test save/load
    config.save("test_config.json")
    loaded_config = SGWCNConfig.load("test_config.json")
    print(f"\nLoaded config matches: {config.to_dict() == loaded_config.to_dict()}")
    
    # Clean up
    os.remove("test_config.json") 
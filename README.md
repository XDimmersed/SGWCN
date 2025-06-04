# Spiking Graph Wavelet Convolution Network (SGWCN)

基于脉冲神经网络的图小波卷积网络，用于3D点云分类任务。结合了DGCNN、GWNN和脉冲神经网络的创新技术，实现了高能效的点云处理。

## 📚 项目概述

SGWCN是一个创新的神经网络架构，专为3D点云分类设计，主要特色包括：

### 核心技术创新
- **局部密度自适应尺度选择**: 使用k近邻距离计算自适应边权重和扩散尺度
- **节点级自适应图小波卷积**: Chebyshev系数变为节点特定的自适应参数
- **双极性LIF神经元**: 采用双阈值机制编码凸/凹特征
- **稀疏脉冲表示**: 实现约100倍的能耗降低

### 架构特点
- **数据流**: [B,N,3] → kNN图 → [B,N,k]边权重 → 小波卷积 → [B,T,N,F]脉冲 → 分类
- **图构建**: 标准高斯核 exp(-dist²/(2σ²)) 配合局部自适应σ
- **脉冲编码**: 批次优先[B,T,N,F]格式，泊松编码
- **优化策略**: 稀疏边表示、递归Chebyshev计算、批次归一化

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone <repository-url>
cd sgwcn

# 安装依赖
pip install -r requirements.txt

# 验证安装
python quick_start.py
```

### 2. 数据准备

确保ModelNet40数据集位于正确位置：
```
data/
└── modelnet40_ply_hdf5_2048/
    ├── ply_data_train0.h5
    ├── ply_data_train1.h5
    ├── ...
    ├── ply_data_test0.h5
    ├── ply_data_test1.h5
    ├── train_files.txt
    ├── test_files.txt
    └── shape_names.txt
```

### 3. 快速测试

运行快速测试脚本验证环境：
```bash
python quick_start.py
```

该脚本将测试：
- 数据加载功能
- 模型创建和前向传播
- 简短的训练演示

## 🎯 训练模型

### 基础训练

```bash
# 使用默认配置训练
python train_sgwcn.py

# 使用快速调试配置
python train_sgwcn.py --config fast_debug

# 使用高性能配置
python train_sgwcn.py --config high_performance

# 使用节能配置
python train_sgwcn.py --config energy_efficient
```

### 自定义训练参数

```bash
# 自定义批次大小和学习率
python train_sgwcn.py --batch_size 16 --learning_rate 0.0005

# 指定训练轮数和设备
python train_sgwcn.py --num_epochs 100 --device cuda

# 从检查点恢复训练
python train_sgwcn.py --resume checkpoints/checkpoint_best.pth
```

### 配置预设说明

- **default**: 标准配置，平衡性能和速度
- **fast_debug**: 快速调试，小模型小数据
- **high_performance**: 高性能配置，追求最佳准确率
- **energy_efficient**: 节能配置，优化能耗比

## 📊 模型评估

### 基础评估

```bash
# 评估训练好的模型
python evaluate_sgwcn.py checkpoints/checkpoint_best.pth

# 指定配置文件
python evaluate_sgwcn.py checkpoints/checkpoint_best.pth --config configs/config.json

# 快速评估（较少样本）
python evaluate_sgwcn.py checkpoints/checkpoint_best.pth --quick
```

### 评估结果

评估脚本将生成：
- **准确率指标**: 总体准确率、Top-3、Top-5准确率
- **混淆矩阵**: 详细的分类结果可视化
- **性能报告**: 每类别的精确率、召回率、F1分数
- **推理时间**: 平均推理时间和吞吐量统计

结果保存在 `evaluation_results/` 目录下。

## 🔧 项目结构

```
sgwcn/
├── src/                          # 源代码
│   ├── models/                   # 模型定义
│   │   ├── __init__.py
│   │   ├── graph_builder.py      # 图构建模块
│   │   ├── wavelet_conv.py       # 图小波卷积
│   │   ├── spiking_neurons.py    # 脉冲神经元
│   │   └── sgwcn.py             # 主网络
│   ├── utils/                    # 工具函数
│   │   ├── graph_utils.py        # 图处理工具
│   │   └── spike_utils.py        # 脉冲处理工具
│   └── data/                     # 数据加载
│       └── dataset.py            # ModelNet40数据集
├── configs/                      # 配置文件
│   └── sgwcn_config.py          # 训练配置
├── data/                         # 数据目录
│   └── modelnet40_ply_hdf5_2048/ # ModelNet40数据
├── tests/                        # 测试文件
├── train_sgwcn.py               # 训练脚本
├── evaluate_sgwcn.py            # 评估脚本
├── quick_start.py               # 快速测试
├── requirements.txt             # 依赖包
└── README.md                    # 项目说明
```

## 📈 训练监控

### Tensorboard

训练过程可通过Tensorboard监控：
```bash
tensorboard --logdir logs
```

监控指标包括：
- 训练/验证损失
- 训练/验证准确率
- 学习率变化
- 梯度统计

### 日志文件

训练日志保存在 `logs/` 目录：
- `sgwcn_train_YYYYMMDD_HHMMSS.log`: 详细训练日志
- Tensorboard事件文件

## ⚙️ 配置选项

### 模型参数
- `hidden_dims`: 隐藏层维度列表
- `num_time_steps`: 脉冲时间步数
- `k_neighbors`: k近邻数量
- `chebyshev_order`: Chebyshev多项式阶数

### 训练参数
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `num_epochs`: 训练轮数
- `optimizer`: 优化器选择

### 脉冲神经元参数
- `tau_mem`: 膜电位时间常数
- `theta_pos`: 正阈值
- `theta_neg`: 负阈值

完整配置选项请参考 `configs/sgwcn_config.py`。

## 🔬 性能指标

### 预期性能
- **准确率**: 在ModelNet40上达到93.5%
- **能耗**: 相比传统ANN降低约100倍
- **推理速度**: 实时处理能力
- **内存效率**: 稀疏脉冲表示

### 能效分析
- SNN能耗估计: ~nJ级别每样本
- 稀疏度: >90%的稀疏脉冲
- 计算复杂度: O(N·k)线性复杂度

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小批次大小
   python train_sgwcn.py --batch_size 8
   
   # 或使用CPU
   python train_sgwcn.py --device cpu
   ```

2. **数据加载错误**
   ```bash
   # 检查数据路径
   ls -la data/modelnet40_ply_hdf5_2048/
   
   # 运行快速测试
   python quick_start.py
   ```

3. **依赖包问题**
   ```bash
   # 重新安装依赖
   pip install -r requirements.txt --upgrade
   
   # 检查PyTorch版本
   python -c "import torch; print(torch.__version__)"
   ```

### 调试模式

使用快速调试配置进行问题诊断：
```bash
python train_sgwcn.py --config fast_debug --batch_size 2
```

## 📄 引用

如果您使用了本项目，请引用：

```bibtex
@article{sgwcn2024,
  title={Spiking Graph Wavelet Convolution Network for 3D Point Cloud Classification},
  author={[Your Name]},
  journal={[Conference/Journal]},
  year={2024}
}
```

## 📞 支持

如有问题或建议，请：
1. 查看项目Issues
2. 运行 `python quick_start.py` 进行诊断
3. 提交详细的错误报告

## 📝 许可证

本项目采用MIT许可证。详情请参见LICENSE文件。 
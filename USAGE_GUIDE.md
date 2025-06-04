# SGWCN 使用指南

## 🎯 快速开始（5分钟上手）

### 1. 环境检查
```bash
# 检查环境是否正确配置
python run_sgwcn.py check
```

预期输出：
```
🔍 Checking environment...
  ✓ torch
  ✓ torch_geometric  
  ✓ h5py
  ✓ numpy
  ✓ matplotlib
  ✓ Data directory: data/modelnet40_ply_hdf5_2048
    ✓ train_files.txt
    ✓ test_files.txt
    ✓ shape_names.txt
  ✓ src/
  ✓ configs/

✅ Environment check passed!
```

### 2. 快速测试
```bash
# 运行功能测试（约2-3分钟）
python run_sgwcn.py test
```

预期结果：所有测试通过，确认数据加载和模型创建正常。

### 3. 快速训练演示
```bash
# 快速训练演示（约5-10分钟）
python run_sgwcn.py train --config fast_debug
```

## 🚀 完整训练流程

### 训练配置选择

#### 快速调试（推荐新手）
```bash
python run_sgwcn.py train --config fast_debug
```
- 模型：小型网络（32→64维）
- 数据：少量点云（512点）
- 时间：约10-20分钟
- 用途：验证流程，快速迭代

#### 标准训练（推荐一般使用）
```bash
python run_sgwcn.py train --config default
```
- 模型：标准网络（64→128→256维）
- 数据：完整点云（1024点）
- 时间：约2-4小时
- 用途：正常实验和论文复现

#### 高性能训练（追求最佳结果）
```bash
python run_sgwcn.py train --config high_performance
```
- 模型：大型网络（128→256→512维）
- 数据：密集点云（2048点）
- 时间：约6-12小时
- 用途：发表论文，刷榜

#### 节能训练（资源受限）
```bash
python run_sgwcn.py train --config energy_efficient
```
- 模型：优化网络（少时间步，高稀疏度）
- 数据：标准点云（1024点）
- 时间：约1-3小时
- 用途：移动设备，嵌入式应用

### 自定义训练参数

```bash
# 自定义批次大小（显存不够时减小）
python run_sgwcn.py train --batch_size 8

# 自定义学习率
python run_sgwcn.py train --learning_rate 0.0005

# 组合使用
python run_sgwcn.py train --config default --batch_size 16 --num_epochs 100
```

### 训练监控

#### Tensorboard监控
```bash
# 另开终端运行
tensorboard --logdir logs
```
然后访问 http://localhost:6006 查看：
- 训练/验证损失曲线
- 准确率变化
- 学习率调度
- 梯度统计

#### 日志监控
```bash
# 实时查看日志
tail -f logs/sgwcn_train_*.log
```

### 断点续训
```bash
# 从最新检查点恢复
python run_sgwcn.py train --resume checkpoints/checkpoint_latest.pth

# 从最佳模型恢复
python run_sgwcn.py train --resume checkpoints/checkpoint_best.pth
```

## 📊 模型评估

### 完整评估
```bash
# 评估最佳模型
python run_sgwcn.py eval checkpoints/checkpoint_best.pth
```

生成结果：
- `evaluation_results/evaluation_results.json`：详细数值结果
- `evaluation_results/confusion_matrix.png`：混淆矩阵图
- 控制台输出：关键指标摘要

### 快速评估
```bash
# 仅测试准确率（约1-2分钟）
python run_sgwcn.py eval checkpoints/checkpoint_best.pth --quick
```

## 🔧 常见问题解决

### 内存不足
```bash
# 减小批次大小
python run_sgwcn.py train --batch_size 4

# 或使用CPU（慢但稳定）
python run_sgwcn.py train --device cpu

# 减少点云点数
python run_sgwcn.py train --config fast_debug
```

### CUDA问题
```bash
# 检查CUDA状态
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 强制使用CPU
python run_sgwcn.py train --device cpu
```

### 数据加载错误
```bash
# 检查数据完整性
python -c "
import h5py
import os
data_dir = 'data/modelnet40_ply_hdf5_2048'
files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
for file in files:
    try:
        with h5py.File(os.path.join(data_dir, file), 'r') as f:
            print(f'✓ {file}: {list(f.keys())}')
    except Exception as e:
        print(f'❌ {file}: {e}')
"
```

### 依赖包问题
```bash
# 重新安装核心依赖
pip install torch torchvision torch-geometric h5py numpy --upgrade

# 检查torch-geometric版本兼容性
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

## 📈 性能优化建议

### GPU训练优化
1. **批次大小**：尽量使用2的幂次（4, 8, 16, 32）
2. **混合精度**：默认开启，可节省50%显存
3. **编译模型**：PyTorch 2.0+可设置 `compile_model=True`

### 超参数调优
1. **学习率**：从0.001开始，可尝试0.0005或0.002
2. **时间步数**：4-16之间，越多越准确但越慢
3. **k邻居数**：10-30之间，点云密度高时可增大

### 数据增强
- 训练时默认开启旋转、缩放、抖动
- 评估时关闭所有增强
- 可在配置中调整增强强度

## 🎓 进阶使用

### 自定义配置文件
```python
# 创建 my_config.py
from configs.sgwcn_config import SGWCNConfig

config = SGWCNConfig(
    hidden_dims=[128, 256, 512],  # 自定义网络结构
    num_time_steps=8,             # 自定义时间步
    k_neighbors=25,               # 自定义邻居数
    learning_rate=0.0008,         # 自定义学习率
    batch_size=20                 # 自定义批次
)

# 保存配置
config.save('configs/my_config.json')
```

```bash
# 使用自定义配置
python train_sgwcn.py --config_file configs/my_config.json
```

### 模型分析
```python
# 加载模型进行分析
from src.models import SGWCNClassifier
import torch

model = SGWCNClassifier(num_points=1024)
dummy_input = torch.randn(1, 1024, 3)

# 详细分析
analysis = model.forward_with_analysis(dummy_input)
print(f"Energy: {analysis['energy_consumption']:.2e} J")
print(f"Spikes: {analysis['total_spikes']}")
```

### 导出ONNX模型
```python
import torch
from src.models import SGWCNClassifier

# 加载训练好的模型
model = SGWCNClassifier.load_from_checkpoint('checkpoints/best_model.pth')
model.eval()

# 导出ONNX
dummy_input = torch.randn(1, 1024, 3)
torch.onnx.export(
    model, dummy_input, 'sgwcn_model.onnx',
    export_params=True,
    opset_version=11,
    input_names=['point_cloud'],
    output_names=['logits']
)
```

## 📋 检查清单

训练前检查：
- [ ] 环境检查通过：`python run_sgwcn.py check`
- [ ] 快速测试通过：`python run_sgwcn.py test`
- [ ] 确认训练配置：批次大小适合显存
- [ ] 设置监控：启动tensorboard

训练中监控：
- [ ] 损失正常下降
- [ ] 准确率稳步提升
- [ ] 无内存溢出错误
- [ ] 模型按时保存

训练后验证：
- [ ] 最佳模型存在：`checkpoints/checkpoint_best.pth`
- [ ] 完整评估：`python run_sgwcn.py eval`
- [ ] 结果合理：准确率>85%（快速调试>70%）

## 🚀 下一步

1. **优化模型**：尝试不同的网络结构和超参数
2. **分析结果**：研究混淆矩阵，找出难分类的类别
3. **拓展应用**：将模型应用到其他点云数据集
4. **发布部署**：导出模型用于实际应用

祝您使用愉快！🎉 

python train_sgwcn.py --config rtx4090_optimized
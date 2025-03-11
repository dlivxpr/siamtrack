# SiamTrack

SiamTrack是一个基于PySOT修改的孪生网络单目标跟踪框架。该项目提供了一个灵活、高效的视觉目标跟踪实现，支持多种骨干网络和跟踪策略。

## 项目特点

- 基于孪生网络的单目标跟踪框架
- 支持多种骨干网络（ResNet、AlexNet、MobileNetV2等）
- 提供完整的训练、测试和评估工具
- 包含可视化演示和CAM（类激活映射）分析工具
- 易于扩展的模块化设计

## 环境要求

- Python 3.6+
- PyTorch 1.0+
- CUDA 9.0+（推荐用于GPU加速）
- 其他依赖项（见安装说明）

## 安装

1. 克隆仓库
```bash
git clone https://github.com/yourusername/siamtrack.git
cd siamtrack
```

2. 创建并激活虚拟环境（可选但推荐）
```bash
conda create -n siamtrack python=3.7
conda activate siamtrack
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

## 项目结构

```
siamtrack/
├── siamese/           # 核心跟踪框架
│   ├── core/          # 配置和核心功能
│   ├── datasets/      # 数据集接口
│   ├── models/        # 网络模型定义
│   │   ├── backbone/  # 骨干网络（ResNet、AlexNet等）
│   │   ├── head/      # 网络头部
│   │   └── neck/      # 网络颈部
│   ├── tracker/       # 跟踪器实现
│   └── utils/         # 工具函数
├── toolkit/           # 评估工具包
├── tools/             # 训练和测试工具
│   ├── train.py       # 训练脚本
│   ├── test.py        # 测试脚本
│   ├── eval.py        # 评估脚本
│   └── demo.py        # 演示脚本
├── demo/              # 演示视频
├── experiments/       # 实验配置
├── pretrained_models/ # 预训练模型
└── CAM/               # 类激活映射分析
```

## 使用方法

### 训练

使用以下命令开始训练：

```bash
cd tools
python train.py --config ../experiments/config.yaml
```

### 测试

在测试数据集上评估模型：

```bash
cd tools
python test.py --config ../experiments/config.yaml --snapshot model.pth
```

### 演示

使用预训练模型进行视频目标跟踪演示：

```bash
cd tools
python demo.py --config ../experiments/config.yaml --snapshot model.pth --video_name demo/video.mp4
```

## 支持的模型

- 骨干网络：
  - ResNet (18, 34, 50)
  - AlexNet
  - MobileNetV2

## 引用

如果您在研究中使用了本项目，请考虑引用：

```
@inproceedings{your-reference,
  title={Your Paper Title},
  author={Your Name},
  booktitle={Conference},
  year={Year}
}
```

## 许可证

本项目采用 Apache 2.0 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 致谢

本项目基于 [PySOT](https://github.com/STVIR/pysot) 修改，感谢原作者的贡献。 
# SiamTrack

A Siamese network-based single object tracking framework modified from PySOT. This project provides a flexible and efficient visual object tracking implementation that supports various backbone networks and tracking strategies.

[中文文档](README_CN.md)

## Features

- Siamese network-based single object tracking framework
- Support for multiple backbone networks (ResNet, AlexNet, MobileNetV2, etc.)
- Complete training, testing, and evaluation tools
- Visualization demos and CAM (Class Activation Mapping) analysis tools
- Easily extensible modular design

## Requirements

- Python 3.6+
- PyTorch 1.0+
- CUDA 9.0+ (recommended for GPU acceleration)
- Other dependencies (see installation instructions)

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/siamtrack.git
cd siamtrack
```

2. Create and activate a virtual environment (optional but recommended)
```bash
conda create -n siamtrack python=3.7
conda activate siamtrack
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Project Structure

```
siamtrack/
├── siamese/           # Core tracking framework
│   ├── core/          # Configuration and core functionality
│   ├── datasets/      # Dataset interfaces
│   ├── models/        # Network model definitions
│   │   ├── backbone/  # Backbone networks (ResNet, AlexNet, etc.)
│   │   ├── head/      # Network heads
│   │   └── neck/      # Network necks
│   ├── tracker/       # Tracker implementations
│   └── utils/         # Utility functions
├── toolkit/           # Evaluation toolkit
├── tools/             # Training and testing tools
│   ├── train.py       # Training script
│   ├── test.py        # Testing script
│   ├── eval.py        # Evaluation script
│   └── demo.py        # Demo script
├── demo/              # Demo videos
├── experiments/       # Experiment configurations
├── pretrained_models/ # Pretrained models
└── CAM/               # Class Activation Mapping analysis
```

## Usage

### Training

Start training with the following command:

```bash
cd tools
python train.py --config ../experiments/config.yaml
```

### Testing

Evaluate the model on test datasets:

```bash
cd tools
python test.py --config ../experiments/config.yaml --snapshot model.pth
```

### Demo

Run a video object tracking demo using a pretrained model:

```bash
cd tools
python demo.py --config ../experiments/config.yaml --snapshot model.pth --video_name demo/video.mp4
```

## Supported Models

- Backbone Networks:
  - ResNet (18, 34, 50)
  - AlexNet
  - MobileNetV2

## Citation

If you use this project in your research, please consider citing:

```
@inproceedings{your-reference,
  title={Your Paper Title},
  author={Your Name},
  booktitle={Conference},
  year={Year}
}
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project is modified based on [PySOT](https://github.com/STVIR/pysot). We thank the original authors for their contributions.

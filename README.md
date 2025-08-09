# CompressedVQA-HDR

<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=sunwei925/CompressedVQA-HDR)
[![GitHub stars](https://img.shields.io/github/stars/sunwei925/CompressedVQA-HDR)](https://github.com/sunwei925/CompressedVQA-HDR)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-brightgreen?logo=PyTorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/sunwei925/CompressedVQA-HDR)
[![arXiv](https://img.shields.io/badge/arXiv-2507.15709-red?logo=arXiv&label=arXiv)](https://arxiv.org/pdf/2507.11900)

**🏆 🥇 Winner Solution for the FR Track and Second Solution for the NR Track of [ICME 2025 Generalizable HDR and SDR Video Quality Measurement Grand Challenge](https://sites.google.com/view/icme25-vqm-gc/home?authuser=0)**

*Official Implementation of "**CompressedVQA-HDR: Generalized Full-reference and No-reference Quality Assessment Models for Compressed High Dynamic Range Videos**"*

[📖 Paper](https://arxiv.org/pdf/2507.11900)  | [📊 Challenge Results](https://sites.google.com/view/icme25-vqm-gc/home?authuser=0)

</div>

## 🎯 Abstract

Video compression is a fundamental process in modern multimedia systems, essential for efficient storage and transmission while maintaining perceptual quality. The evaluation of compressed video quality is critical for optimizing compression algorithms and ensuring satisfactory user experience. However, existing compressed video quality assessment (VQA) methods often exhibit limited generalization capabilities when confronted with diverse video content, particularly high dynamic range (HDR) videos that present unique challenges due to their extended luminance range and enhanced color gamut.

This repository presents **CompressedVQA-HDR**, a comprehensive VQA framework specifically designed to address the challenges of HDR video quality assessment. Our approach leverages state-of-the-art deep learning architectures: the Swin Transformer for full-reference (FR) assessment and SigLip 2 for no-reference (NR) assessment. The FR model employs intermediate-layer features from the Swin Transformer to compute deep structural and textural similarities between reference and distorted frames. The NR model extracts global mean features from SigLip 2's final-layer feature maps as quality-aware representations.

To overcome the scarcity of HDR training data, we implement a sophisticated training strategy: the FR model undergoes pre-training on large-scale standard dynamic range (SDR) VQA datasets followed by fine-tuning on the HDRSDR-VQA dataset, while the NR model employs an iterative mixed-dataset training approach across multiple compressed VQA datasets before fine-tuning on HDR content.

Our experimental results demonstrate that CompressedVQA-HDR achieves state-of-the-art performance compared to existing FR and NR VQA models. Notably, **CompressedVQA-HDR-FR secured first place in the FR track and second place in the NR track** of the Generalizable HDR & SDR Video Quality Measurement Grand Challenge at IEEE ICME 2025.

## 📦 Installation

### Prerequisites

- Python 3.9+
- PyTorch 1.13+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/sunwei925/CompressedVQA-HDR.git
cd CompressedVQA-HDR

# Create and activate conda environment
conda create -n HDRVQA python=3.9
conda activate HDRVQA

# Install dependencies
pip install -r requirements.txt
```

## 🧪 Usage

### Pre-trained Models

Download the pre-trained models and place them in the `ckpts/` directory:

- **FR Model**: [Model Weights](https://www.dropbox.com/scl/fi/6745joi51g3fuubg2n87i/FR_HDR_VQA.pth?rlkey=atdpom6x6lmosk9tjqijbm44r&st=qcyio5gj&dl=0) | [Model Profile](https://www.dropbox.com/scl/fi/3po4q1e3ojfmvs6f83ow6/FR_HDR_VQA.npy?rlkey=pqb6jvzj2g2qdjpt7tnq8uueo&st=x9lqun6p&dl=0)
- **NR Model**: [Model Weights](https://www.dropbox.com/scl/fi/t9auox7p47yjf0crwxyso/NR_HDR_VQA.pth?rlkey=3hpgvoiq484lt80moh9j6tmkx&st=0q9ofeep&dl=0) | [Model Profile](https://www.dropbox.com/scl/fi/ezsv3buh353ny3y71y4wk/NR_HDR_VQA.npy?rlkey=domijnhpjfsjp9tyvvyfp5zf9&st=3u9734nt&dl=0)

### Full-Reference (FR) Video Quality Assessment

```bash
cd FR

# Evaluate HDR video quality
CUDA_VISIBLE_DEVICES=0 python VQA_FR.py \
    --distorted <path_to_distorted_video> \
    --reference <path_to_reference_video> \
    --model_path ckpts/FR_HDR_VQA.pth \
    --profile_path ckpts/FR_HDR_VQA.npy
```

### No-Reference (NR) Video Quality Assessment

```bash
cd NR

# Evaluate video quality without reference
CUDA_VISIBLE_DEVICES=0 python VQA_NR.py \
    --distorted <path_to_distorted_video> \
    --model_path ckpts/NR_HDR_VQA.pth \
    --profile_path ckpts/NR_HDR_VQA.npy
```


### Supported Formats

- Video: MP4, AVI, MOV
- Resolution: Up to 4K (3840×2160)
- Color Space: HDR10, SDR
- Frame Rate: Variable (automatically detected)

## 📚 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{sun2025compressedvqa,
  title={CompressedVQA-HDR: Generalized Full-reference and No-reference Quality Assessment Models for Compressed High Dynamic Range Videos},
  author={Sun, Wei and Cao, Linhan and Fu, Kang and Zhu, Dandan and Jia, Jun and Hu, Menghan and Min, Xiongkuo and Zhai, Guangtao},
  journal={arXiv preprint arXiv:2507.11900},
  year={2025}
}
```


## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

</div>
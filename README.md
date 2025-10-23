<div align="center">
<img src="assets/UMGen_Logo.png" width="80">
<!-- <h1>UMGen</h1> -->

### [Generating Multimodal Driving Scenes via Next-Scene Prediction](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Generating_Multimodal_Driving_Scenes_via_Next-Scene_Prediction_CVPR_2025_paper.pdf)

[Yanhao Wu](https://yanhaowu.github.io/UMGen)<sup>1,2</sup>, [Haoyang Zhang](https://scholar.google.com.hk/citations?user=PlMpgeIAAAAJ&hl=zh-CN&oi=ao)<sup>2</sup>, [Tianwei Lin](https://wzmsltw.github.io/)<sup>2</sup>, [Lichao Huang](https://scholar.google.com/citations?user=F2e_jZMAAAAJ&hl=en&oi=ao/)<sup>2</sup>, 

[Shujie Luo](https://scholar.google.com.hk/citations?user=BDaj_esAAAAJ&hl=zh-CN&oi=ao/)<sup>2</sup>, [Rui Wu](https://scholar.google.com.hk/citations?user=Z_ZkkbEAAAAJ&hl=zh-CN&oi=ao/)<sup>2</sup>, [Congpei Qiu](https://congpeiqiu.github.io)<sup>1</sup>, [Wei Ke](https://gr.xjtu.edu.cn/en/web/wei.ke/home/)<sup>1</sup>, [Tong Zhang](https://scholar.google.com/citations?user=kCy8JG8AAAAJ&hl=en&oi=ao)<sup>3, 4</sup>,
 
<sup>1</sup> Xi'an Jiaotong University, <sup>2</sup> Horizon Robotics, <sup>3</sup> EPFL, <sup>4</sup> University of Chinese Academy of Sciences

Accepted to CVPR 2025

[![UMGen](https://img.shields.io/badge/ProjectPage-UMGen-blue)](https://yanhaowu.github.io/UMGen/)&nbsp;
[![Paper](https://img.shields.io/badge/Paper-UMGen-blue)](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Generating_Multimodal_Driving_Scenes_via_Next-Scene_Prediction_CVPR_2025_paper.pdf)&nbsp;
</div>

## 🚀 Quick Start
### Set up a new virtual environment
```bash
conda create -n UMGen python=3.8 -y
conda activate UMGen
```
### Install dependency packpages
```bash
UMGen_path="path/to/UMGen"
cd ${UMGen_path}
pip3 install --upgrade pip
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -r requirement.txt
```

### Prepare the data
Download the tokenized data and pretrained weights from https://drive.google.com/drive/folders/1rJEVxWNk4MH_FPdqUMgdjV_PHwKJMS-3?usp=sharing

The directory structure should be:
```bash
UMGen/
├── data
│   ├── controlled_scenes/
|       ├── XX
│   ├── tokenized_origin_scenes/
│       ├── XX
|   ├── weights/
│       ├── image_var.tar
|       ├── map_vae.ckpt
|       ├── UMGen_Large.pt
└── projects/
```


## ⚙️ Inference Usage
### 🚀 Infer Future Frames Freely  
Generate future frames automatically without any external control signals.
```bash
python projects/tools/evaluate.py --infer_task video --set_num_new_frames 30
```

### 🎛️ Infer Future Frames with Control
Generate future frames under specific control constraints, such as predefined trajectories or object behavior control.
```bash
python projects/tools/evaluate.py --infer_task control --set_num_new_frames 30
```

---

## 🧩 To-Do List

- [ ] Release more **tokenized scene data**
- [ ] Release the **code for obtaining scene tokens** using the VAE models
- [ ] Release the **diffusion code** to enhance the videos

---

## 📬 Contact
For any questions or collaborations, feel free to contact me : )
📧 **[wuyanhao@stu.xjtu.edu.cn](mailto:wuyanhao@stu.xjtu.edu.cn)**

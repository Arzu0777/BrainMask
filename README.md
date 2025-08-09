🧠 BrainMask
Brain Tumor Segmentation using Attention U-Net with Squeeze-and-Excitation Blocks


📌 Overview
BrainMask is a deep learning project for automatic brain tumor segmentation from MRI scans, developed using the BraTS 2021 dataset.
It combines Attention U-Net, Residual Connections, and Squeeze-and-Excitation (SE) Blocks to achieve high segmentation accuracy.

🚀 Features
Multi-class tumor segmentation (Whole Tumor, Tumor Core, Enhancing Tumor)

Attention Mechanism for improved focus on tumor regions

Residual + SE Blocks for better feature representation

Grad-CAM Visualizations for model interpretability

High Performance – optimized for Dice and IoU scores above 0.85

📂 Repository Structure
bash
Copy
Edit
BrainMask/
│── main.py                # Training & evaluation pipeline  
│── requirements.txt       # Dependencies  
│── checkpoints/           # Saved model weights  
│── results/               # Predictions, Grad-CAMs, training curves  
│── data/                  # (Optional) Place dataset here  
🛠 Installation
bash
Copy
Edit
git clone https://github.com/username/BrainMask.git
cd NeuroSeg
pip install -r requirements.txt
📊 Training
bash
Copy
Edit
python main.py --train --epochs 100 --batch_size 8

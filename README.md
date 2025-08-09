🧠 BrainMask
Whole Tumor Segmentation in Brain MRI using Attention U-Net with Squeeze-and-Excitation Blocks


📌 Overview
BrainMask is a deep learning project for automatic Whole Tumor segmentation from brain MRI scans, developed on the BraTS 2021 dataset.
The model architecture is based on Attention U-Net with Residual Connections and Squeeze-and-Excitation (SE) Blocks, designed for high accuracy in medical image segmentation.

🚀 Features
Whole Tumor segmentation from brain MRI (single class)

Attention Mechanism for improved tumor localization

Residual + SE Blocks for enhanced feature extraction

High Performance – optimized for Dice and IoU metrics

Supports training, evaluation, and prediction pipelines

📂 Repository Structure
bash
Copy
Edit
BrainMask/
│── main.py                # Training & evaluation pipeline  
│── requirements.txt       # Dependencies  
│── checkpoints/           # Saved model weights  
│── results/               # Predictions, training curves  
│── data/                  # (Optional) Place dataset here  
🛠 Installation
bash
Copy
Edit
git clone https://github.com/username/BrainMask.git
cd BrainMask
pip install -r requirements.txt
📊 Training
bash
Copy
Edit
python main.py --train --epochs 100 --batch_size 8
📈 Results
Metric	Score
Dice	0.87+
IoU	0.85+size 8

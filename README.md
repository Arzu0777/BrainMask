# 🧠 BrainMask
**Whole Tumor Segmentation in Brain MRI using Attention U-Net with Squeeze-and-Excitation Blocks**  

![Training Curves](/results/training_curves.png)  

---

## 📌 Overview  
**BrainMask** is a deep learning project for **automatic Whole Tumor segmentation** from brain MRI scans, developed on the **BraTS 2021** dataset.  
The model architecture is based on **Attention U-Net** with **Residual Connections** and **Squeeze-and-Excitation (SE) Blocks**, designed for high accuracy in medical image segmentation.  

---

## 🚀 Features  
- **Whole Tumor segmentation** from brain MRI (single class)  
- **Attention Mechanism** for improved tumor localization  
- **Residual + SE Blocks** for enhanced feature extraction  
- **High Performance** – optimized for Dice and IoU metrics  
- Supports **training, evaluation, and prediction** pipelines  

---

## 📂 Repository Structure  
```
BrainMask/
│── checkpoints/           # Saved model weights  
│── results/               # Predictions, training curves  
│── data/                  # Place dataset here  
|── scripts/               # Contains all core Python scripts for model training, evaluation, and visualization
│── main.py                # Runs the full pipeline: data loading → model training → evaluation → visualization  
│── requirements.txt       # Dependencies  
```

---

## 🛠 Installation  
```bash
git clone https://github.com/Arzu0777/BrainMask.git
cd BrainMask
pip install -r requirements.txt
```

---

## 📊 Training  
```bash
python main.py --train --epochs 30 --batch_size 8
```

---

## 📈 Results  
| Metric   | Score |
|----------|-------|
| Dice     | 0.87+ |
| IoU      | 0.85+ |

**Sample Prediction:**  
![Prediction](/results/sample_4_pred.png)  
![Prediction](/results/sample_3_pred.png)  

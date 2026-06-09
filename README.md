# Opto-GNN: Protein Fitness Prediction with Graph Neural Networks

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c)

Opto-GNN adalah framework deep learning modular untuk memprediksi fitness varian protein menggunakan kombinasi embeddings dari **ESM-2 (Evolutionary Scale Modeling)** dan **Graph Attention Networks (GATv2)**.

## 📂 Project Structure

```text
opto_gnn_project/
├── configs/          # Configuration files (hyperparameters)
├── data/             # Raw and processed protein datasets
├── outputs/          # Model checkpoints and evaluation plots
├── src/              # Core logic
│   ├── models.py     # GATv2 Architecture
│   └── utils.py      # Custom RankMSELoss and helpers
├── tests_internal.py # Unit tests for model & loss
├── requirements.txt  # Project dependencies
└── README.md         # Documentation
```

## 🚀 Getting Started

### Prerequisites
Pastikan Anda memiliki Python 3.8+ dan `pip` yang terinstal.

### Installation
1. Clone repository:
   ```bash
   git clone https://github.com/ZulmanAG/Opto-GNN.git
   cd Opto-GNN
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🔬 Methodology
- **Node Features**: Menggunakan representasi residu dari model transformer `esm2_t6_8M_UR50D`.
- **Graph Topology**: Dibangun menggunakan contact map yang diekstraksi dari *Attention Maps* layer terakhir ESM-2.
- **Loss Function**: Menggunakan **Hybrid Rank-MSE Loss** untuk menyeimbangkan akurasi nilai absolut dan akurasi peringkat (ranking) antar varian.
- **Ensemble**: Menggabungkan fitur topologi GNN dengan statistik mutasi menggunakan gradient boosting (LightGBM).

## 📊 Performance
Berdasarkan pengujian internal pada data sintetis:
- **Standalone GNN**: Spearman's ρ ≈ 0.30
- **Hybrid Ensemble**: Spearman's ρ ≈ 0.94, R-squared ≈ 0.92

## 🛠 Testing
Jalankan pengujian internal sebelum deployment:
```bash
python tests_internal.py

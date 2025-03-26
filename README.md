# The Food Hazard Detection Challenge

A collaborative project for [*SemEval 2025 Task 9: The Food Hazard Detection Challenge*](https://food-hazard-detection-semeval-2025.github.io/), leveraging advanced natural language processing (NLP) models. This repository contains implementations, experiments, and reports focused on detecting food hazards based on textual data.

## Team Members
- **Matheus Campos**
- **Daniel Menezes**
- **Matheus Laureano**

## Objective
This project aims to develop an efficient solution for detecting food hazards from text, using **Transformers** and **Natural Language Processing (NLP)** techniques.  
The core approach includes:
- **Data analysis and preprocessing**: Cleaning raw text and engineering additional features.
- **Model training and evaluation**: Developing a DistilBERT-based model with integrated temporal features.

---

## Repository Structure
```plaintext
├── data/                 # Raw data, preprocessed datasets, and preparation scripts
├── notebooks/            # Jupyter notebooks for experimentation and analysis
├── models/               # Model code, including feature integration
├── configs/              # Model and training configurations
├── report/               # Collaborative LaTeX report (Overleaf)
├── results/              # Model outputs, charts, and visualizations
├── scripts/              # Utility scripts for execution and analysis
└── README.md             # Project documentation (this file)
```

---

## Implemented Architecture

The implemented architecture is based on **DistilBERT**, enhanced with extensions that integrate additional (temporal) features into the classification head. Key components include:

### **1. Base Model: DistilBERT**
- The **DistilBERT** pre-trained model generates robust embeddings from descriptive text.

### **2. Custom Classification Head**
The classification head is modified to integrate textual and additional features:
- **Feature Engineering**: Temporal features (e.g., incident date) are normalized and incorporated.
- **Representation Fusion**:
  - Text embeddings are concatenated with the additional features.
  - A **custom cross-attention mechanism** allows the model to dynamically adjust weights between textual and temporal signals.
  - The combined representation is passed through a linear layer to generate logits.
- **Loss Function**: Uses `CrossEntropyLoss` for multi-class classification.
  ![architecture](https://github.com/user-attachments/assets/5304d2d1-2426-4479-955a-787f95ef5cb0)


### **3. Optimization and Regularization**
- **Optimizer**: `AdamW` for improved weight regularization.
- **Scheduler**: Linear decay with warm-up to dynamically adjust the learning rate.
- **Dropout**: Applied after feature fusion to reduce overfitting.

### **Model Data Flow**
1. **Input**:
   - Descriptive text
   - Additional features (normalized and, if applicable, cyclically encoded).
2. **Processing**:
   - Text is transformed into embeddings using DistilBERT.
   - The embeddings are concatenated with the additional features.
3. **Output**:
   - Classification logits for each food hazard category.
---

## Results
### Key Findings:
- **Accuracy**: 0.77
- **Temporal feature integration**: Improved model accuracy by [X]% over the baseline.
- **Visualizations**: See the `results/` folder for detailed charts and metrics.

### Prediction Examples:
- Input: `"Recall Notification: FSIS-033-94 - Sausage contaminated"`
  - Output: `Risk Category: Biological`
- Input: `"Plastic fragments in chicken breast - High risk"`
  - Output: `Risk Category: Foreign Body`

---

## Contributions
Each team member contributed to different aspects of the project:
- **Matheus Campos**: Data engineering, model architecture development.
- **Daniel Menezes**: Validation module, result visualization, and report structuring.
- **Matheus Laureano**: Data preprocessing, notebook experimentation, and training pipeline.

Detailed contributions can be found in the report.

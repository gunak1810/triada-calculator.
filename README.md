
# 🩸 TRIADA-IR: Machine Learning for Insulin Resistance

**A Novel Machine Learning Approach for Detecting Insulin Resistance Using Low-Cost Anthropometric and Metabolic Markers.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/ML-Random%20Forest-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

**TRIADA-IR** is a Machine Learning model designed to screen for **Insulin Resistance (IR)** without requiring expensive insulin tests or comprehensive lipid panels.

Trained on **12,674 adults** from the **NHANES (2009–2018)** dataset, the model uses a **Random Forest Classifier** to predict IR with high accuracy, outperforming traditional linear indices like **TyG** and **METS-IR**.

### 🎯 Objective
To provide a low-cost, scalable screening tool for primary care and low-resource settings using only 5 routinely collected variables.

---

## 🚀 Key Features

* **No Insulin Required:** Eliminates the need for expensive fasting insulin assays (HOMA-IR).
* **No Lipid Panel Required:** Works without Triglycerides or HDL (unlike TyG or METS-IR).
* **High Accuracy:**
    * **TRIADA-IR AUC:** **0.870** 🏆
    * METS-IR AUC: 0.843
    * TyG Index AUC: 0.785
* **Simple Inputs:** Uses Age, Gender, Weight, BMI, and Fasting Glucose.
* **Interactive Web App:** Built with Streamlit for instant risk estimation.

---

## 🔬 Methodology

### Data Source
* **Dataset:** National Health and Nutrition Examination Survey (NHANES)
* **Cycles:** 2009–2018 (5 cycles merged)
* **Sample Size:** N = 12,674 Adults (≥18 years)
* **Ground Truth:** HOMA-IR (Top 25% cutoff used to define Insulin Resistance).

### Model Architecture
* **Algorithm:** Random Forest Classifier (Ensemble Learning)
* **Parameters:** 300 Estimators, Max Depth = 6 (Preventing Overfitting)
* **Validation:** 75/25 Train-Test Split with Stratification.

---

## 🛠️ Installation & Usage

Follow these steps to run the calculator on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/gunak1810/triada-ir.git](https://github.com/gunak1810/triada-ir.git)
cd triada-ir

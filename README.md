# Credit Card Fraud Detection Project

This project explores multiple machine learning and deep learning methods to detect fraudulent credit card transactions using a real-world imbalanced dataset.

## Overview

Credit card fraud poses a major threat to both consumers and financial institutions. Early and accurate fraud detection is essential to mitigate financial losses. In this project, we:

- Analyze and preprocess a heavily imbalanced dataset
- Explore data patterns and perform visualization
- Implement and evaluate baseline classification models
- Apply oversampling using ADASYN to improve minority class detection
- Train a 1D Convolutional Neural Network (CNN) to further enhance performance
- Evaluate all models using metrics suitable for imbalanced data such as ROC AUC, Precision, Recall, and F1 Score

## Dataset

The dataset contains over 280,000 anonymized credit card transactions made by European cardholders in September 2013, including only 492 fraudulent cases (0.172%). Due to its size, we do **not** include the raw dataset in this repository.

🔗 You can download the dataset from Kaggle here:  
[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Project Structure
creditcard-fraud-detection/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Baseline_Models.ipynb
│   ├── 03_ADASYN_Models.ipynb
│   └── 04_CNN_Model.ipynb
│
├── README.md
├── requirements.txt
└── model_outputs/
└── *.csv, *.png (optional artifacts)

## Requirements

To run this project, install the following Python packages:

```bash
pip install -r requirements.txt

Key packages:
	•	numpy
	•	pandas
	•	scikit-learn
	•	imbalanced-learn
	•	matplotlib
	•	seaborn
	•	tensorflow or keras

## Results
	•	Baseline tree models performed well on accuracy but struggled to identify minority class instances.
	•	ADASYN oversampling improved recall and F1 scores significantly.
	•	The 1D CNN model achieved strong performance with high AUC and balanced precision-recall, making it a promising approach for real-world deployment.











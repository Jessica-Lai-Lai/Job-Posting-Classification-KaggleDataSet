# Job Posting Classification & Fraud Detection Project

## Project Overview

This project is based on the Kaggle Job Posting dataset and aims to perform text classification and fraud detection on job postings. It includes both traditional machine learning (e.g., Logistic Regression with TF-IDF/Word2Vec features) and deep learning (e.g., Transformer/BERT fine-tuning) solutions, supporting two main tasks: education level classification and real/fake job detection.

## Directory Structure

```
Job-Posting-Classification-KaggleDataSet/
  ├── classifier.py         # Logistic regression classifier and hyperparameter search
  ├── features.py           # Text feature engineering (tokenization, TF-IDF, Word2Vec, etc.)
  ├── helper.py             # Helper functions for data distribution visualization
  ├── job_classification.py # Main pipeline for traditional ML (data, features, training, evaluation)
  ├── job_classifier.py     # Transformer-based deep learning classifier
  ├── word2vec.py           # Word2Vec training and hyperparameter tuning
  ├── test.py               # Simple script for model loading and testing
  ├── Task1.ipynb           # Data exploration and analysis (EDA) notebook
  ├── Task4.ipynb           # BERT fine-tuning and experiments notebook
  └── data/
      └── sample_data.csv   # Sample data
```

## Environment Requirements

- Python 3.7+
- Main dependencies:
  - numpy, pandas, scikit-learn, gensim, nltk, torch, tqdm, matplotlib, seaborn, wordcloud
  - For deep learning: transformers, datasets, evaluate

Install dependencies (adjust as needed):
```bash
pip install numpy pandas scikit-learn gensim nltk torch tqdm matplotlib seaborn wordcloud transformers datasets evaluate
```

## Data Description

- Dataset source: [Kaggle Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- Main fields include: job description, company profile, required education, and a fraud label (fraudulent), etc.

## Main Functional Modules

### 1. Data Exploration & Analysis (`Task1.ipynb`)
- Analyze the structure, missing values, class distribution, text length, word clouds, etc. of the raw data.
- Provides feature selection and data understanding for modeling.

### 2. Traditional Machine Learning Pipeline (`job_classification.py` etc.)
- Data preprocessing and splitting (train/val/test, class balance)
- Feature engineering: TF-IDF, Word2Vec
- Classification model: Logistic Regression
- Hyperparameter tuning: grid search, Bayesian optimization, cross-validation
- Performance evaluation: accuracy, etc.

### 3. Deep Learning Text Classification (`job_classifier.py`, `Task4.ipynb`)
- Transformer-based text classifier implemented with PyTorch
- BERT/DistilBERT fine-tuning (implemented in notebook)
- Supports multi-task (education level classification, real/fake job detection)

### 4. Helper Tools
- Data distribution visualization (`helper.py`)
- Model saving and loading (`test.py`)

## Quick Start

1. Download and unzip the Kaggle dataset into the `data/` directory, or use `sample_data.csv` for testing.
2. Run `job_classification.py` for traditional machine learning training and evaluation:
   ```bash
   python job_classification.py
   ```
3. For deep learning experiments, refer to `Task4.ipynb` and run in Jupyter/Colab.

## Results & Conclusion

- With feature engineering and model tuning, the system can effectively distinguish job postings with different education requirements and detect fraudulent jobs.
- Deep learning methods (such as BERT fine-tuning) perform better on text classification tasks, especially for large-scale data and complex semantics.

## Acknowledgements

- Dataset provided by Kaggle.
- Thanks to open-source tools such as scikit-learn, gensim, and transformers.

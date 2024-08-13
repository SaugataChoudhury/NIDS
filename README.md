# NIDS
# Intrusion Detection System using Machine Learning

This project aims to build an Intrusion Detection System (IDS) using machine learning algorithms to classify network traffic as benign or malicious. The CICIDS2017 dataset is used for training and evaluation.

## Steps

1. **Data Preparation:**
   - Download the CICIDS2017 dataset from Kaggle.
   - Load and preprocess the data, handling missing values and converting data types.
   - Perform undersampling to balance the dataset.
   - Standardize the features using z-score normalization.

2. **Feature Selection:**
   - Identify and remove highly correlated features.

3. **Model Training and Evaluation:**
   - Split the data into training and testing sets.
   - Train various machine learning models, including:
     - AdaBoost
     - Random Forest
     - K-Nearest Neighbors (KNN)
     - Decision Tree
     - Support Vector Machine (SVM)
   - Evaluate the models using cross-validation and metrics such as accuracy, confusion matrix, and classification report.

4. **Visualization:**
   - Generate plots to visualize the performance of different models, including:
     - Features vs Accuracy
     - Accuracy vs Classifier
     - Accuracy vs False Positive/Negative

## Requirements

- Python 3
- Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn, imblearn

## Usage

1. Install the required libraries using pip:
2. Download the CICIDS2017 dataset and place it in the appropriate directory.
3. Run the Jupyter Notebook to execute the code.

## Results

The project evaluates the performance of different machine learning models for intrusion detection. The results are presented in terms of accuracy, confusion matrices, and classification reports. The visualizations provide insights into the relationship between features, accuracy, and error rates.

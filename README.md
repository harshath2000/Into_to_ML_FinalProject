## README.md

# Age Prediction from Voice Data

This repository contains the code and data for the project on age group prediction using voice data.This research project delves into the realm of age group prediction through the analysis of voice data, utilizing a comprehensive set of audio features extracted from recordings. The objective is to discern age-related patterns within the extracted features, thereby enabling accurate age prediction. Two distinct methodologies, classical machine learning (ML) and neural networks, are explored to ascertain their effectiveness in this task.

For classical ML, a total of 23 audio features including "spectral centroid," "spectral bandwidth," "spectral_rolloff," "mfcc1" to "mfcc20," "Chroma Feature," "Spectral Contrast," "Tonnetz," and "RMS Energy" are extracted from the voice recordings. These features capture essential characteristics of the audio signals and form the basis for age prediction models. Logistic regression, Support Vector Machines (SVC), and k-Nearest Neighbors (KNN) are implemented and evaluated as part of the classical ML approach.

 This expanded feature set aims to capture intricate nuances within the voice data for enhanced predictive capabilities. Feedforward Neural Networks (FNN) and Convolutional Neural Networks (CNN) are employed as neural network architectures to uncover complex patterns and relationships within the data.

The project evaluates and compares the performance of these methodologies, assessing their accuracy, precision, and recall in predicting age groups from voice data. Insights gained from this study contribute to my understanding of the efficacy of classical ML and neural networks in handling age prediction tasks based on audio features. The project employs both classical machine learning (ML) and neural network approaches to predict the age group of individuals based on audio features extracted from their voice recordings.

## Project Structure

- **Data Preprocessing**: The preprocessing is conducted using `cv-valid-test.csv` and `test_extracted_features.csv` from the dataset.
- **Classical Machine Learning Models**: Models are trained using `train_dataframe.csv`.
- **Preprocessing** : The preprocessing is conducted using `cv-valid-test.csv` from the dataset.
- **Feedforward Neural Network (FNN) and Convolutional Neural Network (CNN)**: These models are trained using `train_dataframe.csv` and `test_dataframe.csv`.

## Feature Extraction

The feature extraction process utilizes the Librosa library to extract the following audio features:
- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff
- Mel-frequency Cepstral Coefficients (MFCCs)
- Chroma Features
- Spectral Contrast
- Tonnetz
- Root Mean Square Energy (RMSE)

## Classical Machine Learning Models

### Logistic Regression
- **Complexity**: Low
- **Accuracy**: 36%
- **F1 Score (Macro avg)**: 16%
- **F1 Score (Weighted avg)**: 29%
- **Observations**: Logistic regression shows the poorest performance among the models, failing to capture complex patterns necessary for accurate age prediction.

### Support Vector Classifier (SVC) with RBF Kernel
- **Complexity**: Medium
- **Accuracy**: 78%
- **F1 Score (Macro avg)**: 80%
- **F1 Score (Weighted avg)**: 79%
- **Observations**: The SVC with RBF kernel outperforms logistic regression by modeling non-linear relationships effectively, resulting in higher accuracy and F1 scores.

### K-Nearest Neighbors (KNN)
- **Complexity**: Medium
- **Accuracy**: 85%
- **F1 Score (Macro avg)**: 85%
- **F1 Score (Weighted avg)**: 85%
- **Observations**: KNN achieves the highest efficiency among classical ML models, demonstrating strong ability to classify age groups accurately based on voice features.

## Neural Network Models

### Feedforward Neural Network (FNN)
- **Complexity**: High
- **Accuracy**: 72.70%
- **F1 Score (Macro avg)**: 75%
- **F1 Score (Weighted avg)**: 73%
- **Observations**: FNNs capture complex relationships effectively but do not outperform the best classical ML models like KNN. However, they still demonstrate strong predictive capabilities.

### Convolutional Neural Network (CNN)
- **Complexity**: High
- **Accuracy**: 73%
- **F1 Score (Macro avg)**: 78%
- **F1 Score (Weighted avg)**: 73%
- **Observations**: CNNs perform similarly to FNNs, with their hierarchical approach proving effective for modeling intricate audio patterns.

## Results

The table below compares the performance of the models:

| Model                        | Accuracy | F1 Score (Macro avg) | F1 Score (Weighted avg) |
|------------------------------|----------|----------------------|-------------------------|
| Logistic Regression          | 36%      | 16%                  | 29%                     |
| SVC with RBF Kernel          | 78%      | 80%                  | 79%                     |
| K-Nearest Neighbors (KNN)    | 85%      | 85%                  | 85%                     |
| Feedforward Neural Network   | 72.70%   | 75%                  | 73%                     |
| Convolutional Neural Network | 73%      | 78%                  | 73%                     |


For more details, refer to the (Final_Report.pdf).

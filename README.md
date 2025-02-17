# MNIST-Classification-using-MLE-PCA-FDA-and-Discriminant-Analysis
# Handwritten Digit Classification using MNIST (Classes 0, 1, 2)

This project implements a complete classification pipeline for handwritten digit images from the MNIST dataset, focusing on classes 0, 1, and 2. The pipeline includes data loading, preprocessing, dimensionality reduction, and classification using Maximum Likelihood Estimation (MLE), Principal Component Analysis (PCA), Fisher's Discriminant Analysis (FDA), and Discriminant Analysis (LDA/QDA). 

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
    - [Maximum Likelihood Estimation (MLE)](#maximum-likelihood-estimation-mle)
    - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
    - [Fisher's Discriminant Analysis (FDA)](#fishers-discriminant-analysis-fda)
    - [Discriminant Analysis](#discriminant-analysis)
4. [Experimental Results](#experimental-results)
5. [Conclusion](#conclusion)
6. [Future Work](#future-work)
7. [Installation](#installation)
8. [Usage](#usage)

## Introduction

This report implements a complete classification pipeline for handwritten digit images using only classes 0, 1, and 2 from the MNIST dataset. The steps include:

- **Data Loading and Preprocessing:** Downloading the MNIST dataset, normalizing the data, and selecting 100 random samples per class for both training and testing.
- **Maximum Likelihood Estimation (MLE):** Estimating the mean vector and covariance matrix for each class.
- **Principal Component Analysis (PCA):** Reducing the dimensionality of the image space using various settings (95%, 90% variance, and 2 principal components).
- **Fisher’s Discriminant Analysis (FDA):** Computing the FDA projection to further reduce the dimensionality to 2.
- **Discriminant Analysis:** Implementing LDA and QDA on different feature representations.
- **Visualization:** Plotting 2D projections of the data to visualize class separability.

## Dataset

The MNIST dataset is downloaded using `kagglehub.dataset.download("hojjatk/mnist-dataset")`. The dataset contains the following IDX format files:
- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

Custom functions are provided to load the IDX format files into NumPy arrays. The images are normalized to the range [0, 1]. Only images corresponding to classes 0, 1, and 2 are retained, and 100 random samples per class are selected for both training and testing.

## Methodology

### Maximum Likelihood Estimation (MLE)
For each class, the mean vector and covariance matrix are computed from the training data:
- **Mean (µc):** Calculated as the average of all feature vectors for a class.
- **Covariance (Σc):** Computed using the formula:

  [\
  \Sigma_c = \frac{1}{N-1} \sum_{i=1}^{N} (x_i - \mu_c)(x_i - \mu_c)^T
  \]

These estimates provide a basis for the Gaussian class-conditional densities used in QDA.

### Principal Component Analysis (PCA)
PCA is implemented from scratch:
1. Center the training data by subtracting the mean.
2. Compute the covariance matrix.
3. Perform eigen-decomposition of the covariance matrix.
4. Sort eigenvalues in descending order.
5. Select the number of principal components such that 95% or 90% of the total variance is retained. Alternatively, choose the first two components for visualization.

### Fisher's Discriminant Analysis (FDA)
FDA is implemented to compute:
- **Within-class scatter matrix (SW):** Sum of covariance matrices for each class.
- **Between-class scatter matrix (SB):** Computed using differences between class means and the overall mean.
- Solve the generalized eigenvalue problem \( (S_W)^{-1} S_B \) to find the optimal projection matrix.

### Discriminant Analysis
LDA and QDA classifiers are implemented from scratch:
- **LDA (Linear Discriminant Analysis):** Assumes a common covariance matrix for all classes.
- **QDA (Quadratic Discriminant Analysis):** Assumes class-specific covariance matrices.

Both classifiers are evaluated on:
- PCA with 95% variance.
- PCA with 90% variance.
- PCA with only 2 components.
- FDA projection (using PCA 95% features).

## Experimental Results

### FDA Projection Plot
The training data projected onto the 2 FDA dimensions, showing class clusters.

### PCA (2 Components) Plot
The training data plotted using the first two principal components, providing an overview of data variance.

## Conclusion

This report demonstrates a full classification pipeline for MNIST digits (0, 1, and 2) using NumPy and Matplotlib. Key contributions include:
- Data loading and preprocessing.
- Implementation of MLE for class-wise parameter estimation.
- Dimensionality reduction with PCA and FDA.
- Classification using LDA and QDA.
- Visualization of transformed feature spaces.

### Key Observations:
- PCA reduces dimensionality effectively, but FDA enhances class separation.
- The choice of classifier and the number of retained principal components significantly impact classification performance.

## Future Work

Further parameter tuning and testing on a larger subset of the MNIST dataset could improve classification performance. Exploring other dimensionality reduction techniques and classifiers may also be beneficial.

## Installation

To run this project, ensure you have the following dependencies installed:

```bash
pip install numpy matplotlib kagglehub scikit-learn

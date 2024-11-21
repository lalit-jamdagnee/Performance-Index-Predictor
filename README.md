
Here's the live demo of the project: https://performance-index-predictor-vzmfmqqbtnm4kf64f2jmi5.streamlit.app/

# Performance Index Predictor

A machine learning project using an Artificial Neural Network (ANN) model for regression to predict the **Performance Index** based on input features. This project demonstrates the application of deep learning in regression tasks and is implemented using Python and Keras.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Introduction
The **Performance Index Predictor** aims to forecast specific performance metrics given certain input parameters. This project is designed for beginners in deep learning who wish to understand regression problems using ANN.

---

## Features
- Implements a regression model using ANN.
- Supports customizable training parameters such as learning rate, batch size, and epochs.
- Provides clear visualization of the training process with metrics like loss and accuracy.
- Easily extendable for different datasets.

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**: 
  - TensorFlow/Keras for model development
  - NumPy and Pandas for data preprocessing
  - Matplotlib for visualizations

---

## Dataset
The dataset used in this project includes features and corresponding target performance indices. The data undergoes preprocessing steps such as normalization to improve model performance. Replace this section with specific details if a publicly available dataset is used.

---

## Model Architecture
The model is a fully connected ANN with the following layers:
1. Input Layer: Accepts normalized input features.
2. Hidden Layers: Two dense layers with ReLU activation.
3. Output Layer: A single neuron for regression output.

Loss Function: Mean Squared Error (MSE)  
Optimizer: Adam

---

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/lalit-jamdagnee/Performance-Index-Predictor.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python scripts to train and evaluate the model:
   ```bash
   python training.py
   ```

---

## Usage
1. Prepare your dataset and update the script to load it.
2. Customize training parameters as needed.
3. Train the model using the `training.py` script.
4. Evaluate the model and use it for predictions.

---

## Results
The model achieves [insert your performance metrics, e.g., RMSE, MAE] on the test dataset. Visualization of the training process and evaluation metrics is provided in the Jupyter notebook.

---

## Future Enhancements
- Add more advanced regression models like Gradient Boosting or Random Forest for comparison.
- Incorporate cross-validation for better generalization.
- Create a web interface for user-friendly interaction with the model.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


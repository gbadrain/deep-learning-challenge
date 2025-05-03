# Charity Application Success Prediction Model

## Neural Network Model Analysis

This repository contains a deep learning model for predicting whether charity applications will be successful using the Alphabet Soup dataset. The model aims to classify applications as successful (1) or unsuccessful (0) based on various organizational and application features.

## Overview of the Analysis

- **Purpose:** Predict whether a charity application is successful ("IS_SUCCESSFUL") using deep learning.
- **Approach:** Extensive data preprocessing, feature engineering, and a shallow neural network were applied and compared against potential ensemble methods.

## Results

### 1. Data Preprocessing

- **Target Variable:**
  - `IS_SUCCESSFUL` (binary: 1 = successful, 0 = unsuccessful)
- **Feature Variables:**
  - Selected features: `ASK_AMT`, `APPLICATION_TYPE` (with rare types replaced by "Other"), `AFFILIATION`, `CLASSIFICATION` (binned to "Other"), `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, plus one-hot encoded variables.
- **Removed Variables:**
  - `EIN` (unique ID, no predictive power)
  - High-cardinality field `NAME` was binned to reduce noise.

### 2. Compiling, Training, and Evaluating the Model

- **Neural Network Architecture:**
  - **Input:** Preprocessed feature count.
  - **Hidden Layers:**
    - Layer 1: 10 neurons, ReLU
    - Layer 2: 8 neurons, Sigmoid
    - Layer 3: 6 neurons, Sigmoid
  - **Output Layer:**
    - 1 neuron, Sigmoid
  - **Total Trainable Parameters:**
    - 899
- **Training Details:**
  - Loss Function: Binary crossentropy
  - Optimizer: Adam
  - Epochs: 30, Batch Size: 16
  - **Training Accuracy:** ~75.8%–76.0%
- **Test Evaluation:**
  - **Accuracy:** ~75.79%
  - **Precision:** ~71.6%
  - **Recall:** ~90.5%
  - **Loss:** ~0.49

## Model Analysis

### Strengths of Current Approach

1. **Thorough Data Preprocessing**: We've correctly identified and removed non-predictive identifiers like `EIN` and handled high-cardinality features appropriately by binning rare values.

2. **Balanced Architecture**: The neural network employs a gradually decreasing neuron count (10→8→6→1), which is generally good practice to avoid bottlenecks.

3. **Mixed Activation Functions**: Using ReLU in the first layer and Sigmoid in subsequent layers leverages the advantages of both functions.

4. **Solid Recall Performance**: At 90.5%, the model effectively identifies most of the truly successful applications.

### Areas for Improvement

1. **Limited Model Complexity**: With only 899 trainable parameters, the network may be unable to capture more complex patterns in the data.

2. **Precision-Recall Imbalance**: The gap between precision (71.6%) and recall (90.5%) suggests the model is overclassifying positive cases.

3. **Limited Experimentation**: The initial work included experiments with neuron counts and activation functions but would benefit from more systematic exploration.

## Recommendations

### Architecture Refinements

- Consider increasing the complexity of the first hidden layer (e.g., 32 or 64 neurons)
- Experiment with dropout layers (0.2-0.3) between hidden layers to reduce overfitting
- Try Leaky ReLU instead of standard ReLU to handle potential dying neurons

### Feature Engineering

- Create interaction features between related categorical variables
- Explore more granular binning strategies for `ASK_AMT` to capture potential thresholds
- Consider log-transforming skewed numerical features

### Training Optimizations

- Implement early stopping with patience=5 to prevent overfitting
- Try different optimizers (RMSprop, SGD with momentum)
- Experiment with learning rate schedules

### Ensemble Approach

- Ensemble methods could be valuable alternatives to pure neural networks
- Consider a stacked ensemble combining neural network with gradient boosting
- XGBoost with carefully tuned hyperparameters often performs well for this type of binary classification task

## Example Implementation

```python
# Enhanced architecture
model = tf.keras.Sequential([
    # Input layer
    tf.keras.layers.Dense(64, activation='relu', input_dim=input_dim),
    tf.keras.layers.Dropout(0.3),
    
    # Hidden layers
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    
    # Output layer
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile with learning rate schedule
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc')]
)

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# Training with callbacks
history = model.fit(
    X_train, y_train,
    epochs=100,  # Higher max epochs with early stopping
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)
```

## Summary

The current neural network achieved approximately 75.8% accuracy with solid recall (90.5%) and moderate precision (71.6%) using a simple three-hidden-layer architecture. While this provides a good baseline, implementing the recommended improvements has the potential to boost overall performance.

## Project Structure

```
Deep_Learning_Challenge/
├── AlphabetSoupCharity_Optimization.ipynb     # Optimized model implementation
├── AlphabetSoupCharity.ipynb                  # Initial model implementation
├── h5 files/                                  # Saved model files
│   ├── AlphabetSoupCharity_Optimization.h5    # Optimized model weights
│   └── AlphabetSoupCharity.h5                 # Initial model weights
├── Images/                                    # Documentation screenshots
│   ├── AlphabetSoupCharity_Optimization.pdf
│   └── AlphabetSoupCharity.pdf
└── README.md                                  # This documentation file
```

## Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- Pandas
- Scikit-learn
- Matplotlib/Seaborn (for visualization)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/gbadrain/Deep_Learning_Challenge.git
   ```

2. Install required packages:
   ```
   pip install tensorflow pandas scikit-learn matplotlib seaborn
   ```

### Usage

1. Open and run the notebooks in Google Colab or Jupyter:
   - `AlphabetSoupCharity.ipynb` for the initial model implementation
   - `AlphabetSoupCharity_Optimization.ipynb` for the optimized model version

2. The trained models are saved in the `h5 files` directory and can be loaded for inference:
   ```python
   from tensorflow.keras.models import load_model
   
   # Load the optimized model
   model = load_model('h5 files/AlphabetSoupCharity_Optimization.h5')
   
   # Make predictions
   predictions = model.predict(X_test)
   ```

## Future Work

- Implement ensemble methods as recommended in the analysis
- Further feature engineering to improve model performance
- Create a web interface for model inference
- Additional hyperparameter tuning for optimization

## Sources of Help

* Resources from University of Oregon Continuing and Professional Education Data Analytics Boot Camp on 'Supervised Learning.'
* Microsoft Copilot for problem-solving and guidance.

## Acknowledgments

This project was created as part of a professional development challenge inspired by the Module 20 curriculum. Special thanks to the developers of open-source tools and libraries that made this analysis possible.

## Contact

* **Name**: Gurpreet Singh Badrain
* **Role**: Market Research Analyst & Aspiring Data Analyst
* **GitHub**: https://github.com/gbadrain
* **LinkedIn**: http://linkedin.com/in/gurpreet-badrain-b258a0219
* **Email**: gbadrain@gmail.com


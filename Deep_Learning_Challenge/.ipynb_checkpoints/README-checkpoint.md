Neural Network Model Report for Alphabet Soup Funding Prediction
Overview of the Analysis
Purpose: Predict whether a charity application is successful ("IS_SUCCESSFUL") using deep learning.
Approach: Extensive data preprocessing, feature engineering, and a shallow neural network were applied and compared against potential ensemble methods.
Results
1. Data Preprocessing
Target Variable:
IS_SUCCESSFUL (binary: 1 = successful, 0 = unsuccessful)
Feature Variables:
Selected features: ASK_AMT, APPLICATION_TYPE (with rare types replaced by “Other”), AFFILIATION, CLASSIFICATION (binned to “Other”), USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, plus one-hot encoded variables.
Removed Variables:
EIN (unique ID, no predictive power)
High-cardinality field NAME was binned to reduce noise.
> Figure 1: Data Preprocessing Workflow > (Include a schematic showing: Raw Data → Dropping columns → Binning → One-hot encoding → Scaling)
2. Compiling, Training, and Evaluating the Model
Neural Network Architecture:
Input: Preprocessed feature count.
Hidden Layers:
Layer 1: 10 neurons, ReLU
Layer 2: 8 neurons, Sigmoid
Layer 3: 6 neurons, Sigmoid
Output Layer:
1 neuron, Sigmoid
Total Trainable Parameters:
899
> Figure 2: Neural Network Architecture Diagram > (Insert schematic of layers and neuron counts)
Training Details:
Loss Function: Binary crossentropy
Optimizer: Adam
Epochs: 30, Batch Size: 16
Training Accuracy: ~75.8%–76.0%
Test Evaluation:
Accuracy: ~75.79%
Precision: ~71.6%
Recall: ~90.5%
Loss: ~0.49
Performance Improvement Steps:
Replaced low-frequency categorical values with “Other.”
Applied one-hot encoding and standardized numerical features.
Experimented with neuron counts and activation functions to balance complexity and overfitting.
> Figure 3: Training and Evaluation Metrics Output > (Include a screenshot of Colab output showing training metrics and test evaluation.)
Summary
Overall Results: The neural network achieved approximately 75.8% accuracy, with solid recall and moderate precision using a simple three-hidden-layer architecture.
Alternative Recommendation: Ensemble Methods (e.g., Gradient Boosting or Random Forest):
Better capture non-linear interactions and offer robust feature importance insights.
Tuning ensemble models via GridSearchCV (e.g., optimizing n_estimators, max_depth, learning_rate) may yield incremental performance gains.
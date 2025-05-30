# Comparative Analysis of Neural Network Models for Classification Challenge

## Executive Summary

This technical report evaluates three distinct neural network architectures implemented for a binary classification challenge. The models demonstrate an evolutionary approach to solving the problem, progressing from a basic neural network to a more sophisticated hybrid model leveraging ensemble methods for feature selection. Performance metrics indicate incremental improvements across models, with Model 2 achieving the highest accuracy (76.2%) while Model 3 demonstrates superior recall (90.5%) with fewer parameters.

## Model Architecture Analysis

### Model 1: Basic Neural Network
The initial implementation utilizes a straightforward architecture with minimal preprocessing and a simple layer structure.

**Preprocessing Approach:**
- `EIN` (unique ID, no predictive power)
- High-cardinality field `NAME` was binned to reduce noise.
- Standard feature scaling

**Architecture Details:**
- Input Layer: Matches scaled training data dimensions
- Hidden Layer 1: 10 neurons with tanh activation
- Hidden Layer 2: 5 neurons with ReLU activation
- Output Layer: 1 neuron with ReLU activation (suboptimal for classification)

**Model Parameters:**
- Total trainable parameters: 501
Model 1 Parameters Summary
![Screenshot 2025-05-03 at 6 18 39 PM](https://github.com/user-attachments/assets/7ade032a-9450-42ea-8ea0-132f79126124)


**Training Performance:**
Model 1 Accuracy/Loss Curves
![Screenshot 2025-05-03 at 6 19 06 PM](https://github.com/user-attachments/assets/69619d88-1686-480b-a936-12b051fee02f)

**Architectural Limitations:**
- Inappropriate output activation function (ReLU instead of sigmoid)
- Limited model capacity with only 501 parameters
- Absence of regularization techniques

### Model 2: Deep Neural Network
This implementation significantly expands model capacity and introduces more appropriate activation functions.

**Preprocessing Enhancements:**
- EIN column removal
- NAME column binning (reducing cardinality by grouping low-frequency values)

**Architecture Details:**
- Hidden Layer 1: 128 neurons with ReLU activation
- Hidden Layer 2: 64 neurons with sigmoid activation
- Hidden Layer 3: 32 neurons with sigmoid activation
- Output Layer: 1 neuron with sigmoid activation (appropriate for binary classification)

**Model Parameters:**
- Total trainable parameters: 19,969

Model 2 Parameters Summary

![Screenshot 2025-05-03 at 6 28 56 PM](https://github.com/user-attachments/assets/afeb0b09-3ef2-4d7e-8f47-02fc9d96b233)

**Optimization Strategy:**
- Adam optimizer with adaptive learning rates
- Binary cross-entropy loss function
- Multiple evaluation metrics (precision, recall, accuracy)

**Training Performance:**

Model 2 Accuracy/Loss Curves
![Screenshot 2025-05-03 at 6 30 14 PM](https://github.com/user-attachments/assets/4c9fb425-2cbd-4dad-92d2-c651c8050a1f)

**Performance Metrics:**
- Accuracy: 75.6%
- Precision: ~72%
- Recall: ~88%

### Model 3: Hybrid Random Forest + Neural Network
This implementation employs a two-stage approach combining Random Forest for feature selection with a neural network classifier.
![Screenshot 2025-05-03 at 6 32 39 PM](https://github.com/user-attachments/assets/b082a94d-2c26-4094-899d-5002cb1bb2d2)


**Feature Selection Strategy:**
- Random Forest Classifier with 200 estimators
- Feature importance ranking
- Key findings:
  - ASK_AMT identified as dominant feature (34% importance)
  - AFFILIATION_CompanySponsored (9%) and NAME_Other (8%) as secondary contributors
  - Elimination of near-zero importance features

**Neural Network Architecture:**
- Hidden Layer 1: 10 neurons with ReLU activation
- Hidden Layer 2: 8 neurons with sigmoid activation
- Hidden Layer 3: 6 neurons with sigmoid activation
- Output Layer: 1 neuron with sigmoid activation

**Model Parameters:**
- Total trainable parameters: 899

Model 3 Parameters Summary
![Screenshot 2025-05-03 at 6 34 08 PM](https://github.com/user-attachments/assets/34f288ef-2f3d-4cc5-a6d3-8735a513bfb6)

**Training Configuration:**
- Epochs: 30
- Batch size: 16

**Training Performance:**

Model 3 Accuracy/Loss Curves

![Screenshot 2025-05-03 at 6 34 42 PM](https://github.com/user-attachments/assets/13d6ef5c-9f78-48fa-8252-d3279ca437c7)


**Performance Metrics:**
- Accuracy: 75.8%
- Precision: 71.6%
- Recall: 90.5%

## Comparative Performance Analysis

| Metric | Model 1 | Model 2 | Model 3 |
|--------|---------|---------|---------|
| Architecture | 2 hidden layers | 3 hidden layers | RF + 3 hidden layers |
| Parameters | 501 | 19,969 | 899 |
| Output Activation | ReLU | Sigmoid | Sigmoid |
| Accuracy | ~73% | 75.6% | 75.8% |
| Precision | Not reported | ~72% | 71.6% |
| Recall | Not reported | ~88% | 90.5% |
| Feature Engineering | Basic | Moderate | Advanced |

## Critical Evaluation

### Model 1
**Strengths:**
- Computational efficiency due to small parameter count
- Simplicity facilitates interpretation and debugging

**Limitations:**
- Inappropriate output activation function for classification
- Insufficient model capacity for complex feature relationships
- Limited feature engineering

### Model 2
**Strengths:**
- Highest overall accuracy (76.61%)
- Comprehensive feature extraction capability
- Appropriate activation functions

**Limitations:**
- Potential overfitting risk due to high parameter count
- Computational intensity (40x more parameters than Model 1)
- Sigmoid activations in hidden layers may impede gradient flow

### Model 3
**Strengths:**
- Superior feature selection through Random Forest
- Highest recall rate (88.82%)
- Efficient parameter utilization (95% reduction vs Model 2)

**Limitations:**
- Slight accuracy decrease compared to Model 2
- Potential information loss through feature elimination
- Sigmoid activations in middle layers

### An alternative approach would be using Gradient Boosting Machines (GBMs) like XGBoost or LightGBM instead of a hybrid Random Forest + Neural Network model. Here’s why:

**Advantages of Using GBMs**
**Improved Feature Selection**
Unlike traditional Random Forest, GBMs sequentially refine weak models, prioritizing the most impactful features dynamically. This can enhance interpretability while maintaining predictive power.

**Better Handling of Imbalanced Data**
Since recall was a key strength of Model 3, GBMs could further optimize it by focusing on misclassified instances during training, reducing bias toward majority-class samples.

**Reduced Computational Overhead**
While the hybrid approach involves two stages (Random Forest for feature selection + Neural Network for classification), GBMs offer a single-stage, optimized workflow that streamlines both feature importance ranking and classification.

**Robust Performance Without Extensive Hyperparameter Tuning**
Deep learning models, like Model 2, require meticulous fine-tuning (dropout rates, activation functions, learning rates). GBMs, in contrast, achieve strong results with fewer manual adjustments.

## Conclusion

The comparative analysis indicates that while Model 2 achieves the highest accuracy, Model 3 presents the most efficient architecture with superior recall. The optimization path forward likely involves a hybrid approach that combines Model 2's depth with Model 3's feature selection methodology, while incorporating additional regularization techniques to improve generalization. The significant impact of feature selection demonstrated by Model 3 suggests that further refinement of input features may yield greater performance improvements than architectural changes alone.

# Nested Hierarchical Pitman-Yor Languge Model, Unsupervised Segmentation for Python

Fast Cython implementation of the Nested Hierarchical Pitman-Yor Language Model (NHPYLM) for segmentation and classification.

The library provides two models:
 - **NHPYLMModel**: Performs unsupervised segmentation.
 - **NHPYLMClassesModel**: Extends unsupervised segmentation by incorporating classification during inference. Each class is associated with a separate NHPYLM submodel, which independently segments sequences based on its learned structure. During inference, an input sequence is segmented by all submodels, and the most probable model determines the predicted class.
 

The model processes the input string `Thecatquietlyobservedthebirdsoutsidebeforeleapingontothewindowsill.` and outputs the segmented sentence: `The cat quietly observed the birds outside before leaping onto the windowsill.`.
Additionally, the NHPYLMClassesModel can also predict the class of the sentence, for instance `observation`.


## Usage

### NHPYLMModel (Segmentation)

```python
from nhpylm.models import NHPYLMModel

train_x = ["aaaaaaaaa", "aaaabbbbbbaaa", "aaabbbbbcbaaa"]
dev_x = ["aaaaaaabbba", "abaaaaaccaaa", "bbbaaa"]
test_x = ["bbb", "aaaaa", "aaaaaaa"]
epochs = 20

# Init model
model = NHPYLMModel(7, init_d = 0.5, init_theta = 2.0,
                init_a = 6.0, init_b = 0.83333333,
                beta_stops = 1.0, beta_passes = 1.0,
                d_a = 1.0, d_b = 1.0, theta_alpha = 1.0, theta_beta = 1.0)
# Train and Fit model
model.train(train_x, dev_x, epochs, True, True, print_each_nth_iteration=10)


# Predictions
train_segmentation, train_perplexity = model.predict_segments(train_x)
print("Train Perplexity: {}".format(train_perplexity))
print(train_segmentation)
dev_segmentation, dev_perplexity = model.predict_segments(dev_x)
print("Dev Perplexity: {}".format(dev_perplexity))
print(dev_segmentation)
test_segmentation, test_perplexity = model.predict_segments(test_x)
print("Test Perplexity: {}".format(test_perplexity))
print(test_segmentation)
```

### NHPYLMClassesModel (Segmentation & Classification)
Classification model based on the Conditional NHPYLM segmentation.


```python
from nhpylm.models import NHPYLMClassesModel

train_x = ["aaaaaaaaa", "aaaabbbbbbaaa", "aaabbbbbcbaaa"]
train_y = ["class1", "class1", "class2"] 
dev_x = ["aaaaaaabbba", "abaaaaaccaaa", "bbbaaa"]
dev_y = ["class2", "class2", "class1"]
test_x = ["bbb", "aaaaa", "aaaaaaa"]
test_y = ["class2", "class1", "class1"]
epochs = 20

# Init model
model = NHPYLMClassesModel(7, init_d = 0.5, init_theta = 2.0,
                init_a = 6.0, init_b = 0.83333333,
                beta_stops = 1.0, beta_passes = 1.0,
                d_a = 1.0, d_b = 1.0, theta_alpha = 1.0, theta_beta = 1.0)
# Train and Fit model
model.train(train_x, dev_x, train_y, dev_y, epochs, True, True, print_each_nth_iteration=10)


# Predictions
train_segmentation, train_perplexity, train_mode_prediction = model.predict_segments_classes(train_x)
print("Train Perplexity: {}".format(train_perplexity))
print(train_mode_prediction)
print(train_segmentation)
dev_segmentation, dev_perplexity, dev_mode_prediction = model.predict_segments_classes(dev_x)
print("Dev Perplexity: {}".format(dev_perplexity))
print(dev_mode_prediction)
print(dev_segmentation)
test_segmentation, test_perplexity, test_mode_prediction = model.predict_segments_classes(test_x)
print("Test Perplexity: {}".format(test_perplexity))
print(test_mode_prediction)
print(test_segmentation)
```

## Install this package locally

```sh
pip install .
```


## About NHPYLM

NHPYLM predicts the most probable segmentation for input sequences using a nested structure of Hierarchical Pitman-Yor Language Models (HPYLMs). The first HPYLM operates at the segment level, learning distributions of melodic or textual units. The second HPYLM models character-level (or tone-level) distributions, forming the base for segmentation.

Training is performed using Gibbs sampling, iteratively refining segmentations to maximize posterior probabilities. At inference, segmentation is determined via the Viterbi algorithm.

For classification, multiple NHPYLM models are trained per category (e.g., chant modes). An unknown sequence is segmented by all models, and the most probable model assigns the class.

## How to cite

```
Anonymized for Submission
```
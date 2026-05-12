# Module 7 Week A — Lab Evaluation Report

## Dataset

The dataset contains 7,472 app reviews collected from 9 applications.
The sentiment labels are negative, neutral, and positive.
The dataset was split into 80% training and 20% testing using seed 42.

## Model and hyperparameters

- Backbone: distilbert-base-uncased
- Number of labels: 3
- Learning rate: 5e-5
- Epochs: 2
- Batch size: 8
- Max length: 128
- Seed: 42
- Training time: approximately 31 minutes on CPU

## Metrics on the test split

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 0.6375 |
| Macro-F1 | 0.6356 |

### Per-class F1

| Class    | F1  |
| -------- | --- |
| Negative | ... |
| Neutral  | ... |
| Positive | ... |

## Confusion matrix

| True \ Pred | Negative | Neutral | Positive |
| ----------- | -------- | ------- | -------- |
| Negative    | 357      | 123     | 19       |
| Neutral     | 104      | 239     | 120      |
| Positive    | 33       | 143     | 357      |

The model most commonly confuses neutral and positive reviews.
Neutral reviews appear to be the hardest class to classify correctly.

## Three qualitative error examples

### Example 1

- Original sentence: "..."
- Gold label: neutral
- Predicted label: positive
- Predicted probability for gold label: ...

Explanation:
The sentence contains mildly positive wording that likely caused the model to predict positive instead of neutral.

### Example 2

- Original sentence: "..."
- Gold label: negative
- Predicted label: neutral
- Predicted probability for gold label: ...

Explanation:
The review expresses dissatisfaction indirectly without strong negative words, making classification harder.

### Example 3

- Original sentence: "..."
- Gold label: positive
- Predicted label: neutral
- Predicted probability for gold label: ...

Explanation:
The sentence contains mixed sentiment and lacks strong positive indicators.

## Hugging Face Hub model URL

https://huggingface.co/MohammedRaed/m7-app-review-sentiment

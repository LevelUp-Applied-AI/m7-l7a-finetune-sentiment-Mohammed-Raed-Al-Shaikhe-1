# Calibration Analysis

## Reliability diagram interpretation

The reliability diagram shows that the model is moderately over-confident across most confidence ranges. In a perfectly calibrated model, the empirical accuracy bars would closely follow the diagonal reference line. In this diagram, most buckets fall below the diagonal, meaning the model’s predicted confidence is higher than its true accuracy.

For example, the 0.7–0.8 confidence bucket achieved an empirical accuracy of about 0.62, indicating that predictions made with roughly 75% confidence were correct only around 62% of the time. Similarly, the 0.8–0.9 confidence bucket showed accuracy near 0.75, which is still below the ideal calibration line. Even the highest-confidence bucket (0.9–1.0) achieved accuracy around 0.87 instead of matching its predicted confidence.

Lower-confidence buckets also showed lower empirical accuracy, which is expected because uncertain predictions are typically harder classification cases.

Overall, the diagram indicates that the model tends to assign confidence scores that are too high relative to its true correctness rate.

## Expected Calibration Error

The Expected Calibration Error (ECE) for the model is:

**ECE = 0.1018**

This means that, on average, the model’s predicted confidence differs from its actual accuracy by about 10 percentage points. While the classifier achieves reasonable predictive performance overall, its probability estimates are not fully reliable.

An ECE around 0.10 suggests moderate miscalibration. The model can still be useful for sentiment classification tasks, but its confidence scores should not be interpreted as perfectly trustworthy probabilities without additional calibration. In a production environment, this level of miscalibration could cause the system to be overly confident in incorrect predictions.

## A specific calibration pattern

One clear pattern in the reliability diagram is systematic over-confidence in medium-to-high confidence predictions. The model frequently predicts probabilities between 0.6 and 0.9, but the actual accuracies in those buckets are consistently lower than the predicted confidence values.

This pattern likely arose because the model was trained using cross-entropy loss, which focuses on improving classification accuracy but does not explicitly optimize calibration quality. Transformer-based classifiers such as DistilBERT are also known to produce over-confident softmax probabilities, especially when fine-tuned on relatively small or noisy datasets. Additionally, sentiment classification contains many ambiguous examples near class boundaries, where the model may still output strong confidence despite uncertainty.

## A proposed engineering action

One practical production improvement would be to apply temperature scaling on a validation dataset after training. Temperature scaling is a lightweight calibration technique that adjusts the model’s softmax confidence scores without changing the predicted classes.

Another useful improvement would be to introduce confidence-threshold abstention. For example, predictions with confidence below a threshold such as 0.6 could be routed to human review or a fallback system instead of being accepted automatically. This would reduce the risk of relying on incorrect predictions that the model presents with high confidence.

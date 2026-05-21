# Adversarial Analysis

## Per-hypothesis accuracy

| Category | Correct | Incorrect | Accuracy |
|---|---|---|---|
| negation | 5 | 1 | 83.3% |
| lexical_trigger | 2 | 3 | 40.0% |
| domain_shift | 3 | 2 | 60.0% |
| length_extreme | 3 | 2 | 60.0% |
| sarcasm | 1 | 4 | 20.0% |
| other | 5 | 0 | 100.0% |

Overall accuracy: 63.3%

---

## Confirmed hypotheses

The model struggled heavily with sarcasm examples, which confirmed my original hypothesis that the classifier would rely too strongly on positive lexical cues.

For example, row 18 ("Perfect timing for the server to stop working.") was incorrectly predicted as positive even though the intended sentiment was clearly negative. Similarly, row 19 ("Wonderful another update that breaks everything.") and row 20 ("Yeah this bug definitely improved my workflow.") were also classified as positive. These examples suggest that strong positive cue words such as "Perfect," "Wonderful," and "improved" strongly influenced the model even when the surrounding context reversed the meaning.

The lexical-trigger category also caused several failures. Row 7 ("Amazing how quickly the app crashes now.") was predicted as positive despite describing a negative outcome. Row 8 ("The so-called smart feature barely works.") was predicted as neutral instead of negative, suggesting that the model struggled when positive cue words appeared inside negative contexts.

The model also showed some difficulty handling long multi-clause examples. Row 16 was predicted as neutral even though the sentence described multiple negative problems such as excessive battery usage, failed preferences, and unexpected logouts. This suggests that the model sometimes loses sentiment consistency over longer inputs.

---

## Refuted hypotheses

I expected the model to fail more often on negation examples, but it handled most of them correctly. Rows 1, 4, 5, and 26 were all correctly classified as negative despite containing positive trigger words such as "improve," "stable," "recommend," and "faster." This suggests that the fine-tuned model learned at least some sensitivity to negation structures.

I also expected the model to struggle significantly with noisy informal language, but the "other" category achieved perfect accuracy. Examples containing slang, capitalization, and informal phrasing such as row 23 ("BEST UPDATE EVER!!! and now nothing works.") and row 25 ("Login failed lol nice job developers.") were classified correctly. This indicates that the model is relatively robust to noisy app-review style text.

Some domain-shift examples were also handled better than expected. For example, row 12 ("Heavy rain delayed the match for two hours.") and row 28 ("The scientist confirmed the experiment results yesterday.") were both correctly classified as neutral even though they came from sports and scientific-report contexts rather than app reviews.

---

## What the results reveal about the decision boundary

The adversarial results suggest that the model relies heavily on strong lexical sentiment cues when making predictions. Positive words such as "Perfect," "Wonderful," "Amazing," and "improved" frequently dominated the prediction even when the surrounding sentence clearly described negative situations.

The decision boundary also appears more robust to explicit negation than expected. Phrases such as "did not improve" and "cannot recommend" were usually classified correctly, indicating that the model learned some contextual relationships rather than relying only on isolated keywords.

However, sarcasm remains a major weakness because sarcastic sentences often preserve positive surface-level wording while reversing the intended meaning. The model appears to prioritize surface lexical polarity over pragmatic interpretation.

The model additionally showed some uncertainty on long and mixed-sentiment examples, often predicting neutral when the sentence contained multiple clauses or competing signals. This suggests that the classifier struggles when sentiment is distributed across longer contexts instead of being expressed through a single dominant cue.
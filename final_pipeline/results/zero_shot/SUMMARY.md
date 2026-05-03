# Zero-Shot Prompting Results

## Configuration
- **Model**: GPT-4o-mini
- **Temperature**: 0.0 (deterministic)
- **Split**: Test set (272 windows, 57 videos)
- **Aggregation**: Majority voting

---

## Window-Level Results

| Metric | Value |
|--------|-------|
| Accuracy | 65.44% |
| Macro F1 | 0.5637 |

### Confusion Matrix
```
                 Predicted
                 FALL    NO_FALL
  Actual FALL      27       91
  Actual NO_FALL    3      151
```

### Per-Class Metrics
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Fall | 0.9000 | 0.2288 | 0.3649 | 118 |
| No-Fall | 0.6240 | 0.9805 | 0.7626 | 154 |

---

## Video-Level Results

| Metric | Value |
|--------|-------|
| Accuracy | 66.67% |
| Macro F1 | 0.5606 |

### Confusion Matrix
```
                 Predicted
                 FALL    NO_FALL
  Actual FALL       5       19
  Actual NO_FALL    0       33
```

### Per-Class Metrics
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Fall | 1.0000 | 0.2083 | 0.3448 | 24 |
| No-Fall | 0.6346 | 1.0000 | 0.7765 | 33 |

---

## Key Observations

1. **High Precision, Low Recall for Falls**: The model is conservative - when it predicts a fall, it's usually correct (90-100% precision), but it misses many actual falls (20-23% recall).

2. **Bias Toward No-Fall**: The model tends to predict "NO_FALL" more often, leading to:
   - High recall for no-fall class (98-100%)
   - Many false negatives for fall class

3. **Baseline Performance**: This establishes the baseline for comparison:
   - **Window F1 for Falls**: 0.3649
   - **Video F1 for Falls**: 0.3448
   - **Overall Accuracy**: ~66%

---

## Next Steps

Techniques to improve upon this baseline:
1. **Few-Shot Prompting**: Provide examples to guide the model
2. **Chain-of-Thought**: Enable step-by-step reasoning
3. **Self-Consistency**: Multiple samples with voting
4. **RAG**: Retrieve similar cases to improve predictions

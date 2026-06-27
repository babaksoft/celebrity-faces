# Experiment 1 — Establishing a Scratch CNN Baseline

## Goal

Determine whether a small CNN can learn meaningful facial representations from scratch.

**Architecture**

32 → 64 → GAP → Dense

**Observations**

- Training converged slowly.
- Validation accuracy remained unstable.
- Both losses decreased, indicating learning was occurring.
- Performance remained insufficient for deployment.

**Conclusion**

A minimal CNN can learn basic features but lacks sufficient representational capacity for this dataset.

---

# Experiment 2 — Optimizer Comparison

## Goal

Determine whether optimization strategy is the primary performance bottleneck.

**Optimizers**

- Adam
- RMSprop
- SGD + Momentum

**Observations**

Adam

- smoothest convergence
- fastest learning
- highest validation accuracy

RMSprop

- similar convergence
- noisier validation behavior

SGD

- significantly slower
- required more epochs
- lower validation performance

**Conclusion**

Optimizer choice affects convergence speed but does not overcome dataset limitations.

The limiting factor appears to be learned visual features rather than optimization.

---

# Experiment 3 — Regularization

## Goal

Study the effects of

* Dropout
* stronger augmentation

**Observations**

Dropout

* reduced overfitting
* narrowed train-validation gap

Extra augmentation

* increased training difficulty
* delayed overfitting
* improved robustness

Combined

* best balance between learning and generalization
* still limited by dataset complexity

**Conclusion**

Regularization improved generalization but could not compensate for insufficient learned visual representations.

---

# Experiment 4 — TensorFlow vs PyTorch

## Goal

Determine whether framework choice materially affects learning.

**Implementation**

Reimplemented the best TensorFlow configuration using

* PyTorch
* PyTorch Lightning

**Observations**

Learning curves showed very similar behavior.

Both frameworks reached comparable performance.

**Conclusion**

Framework choice had negligible impact.

Model architecture and dataset quality dominate performance.

---

# Overall Conclusions

Across multiple controlled experiments, the following trends consistently emerged.

## What mattered

✓ optimizer choice

✓ augmentation

✓ dropout

✓ architecture depth

---

## What mattered less

framework choice

---

## Biggest limiting factor

The dataset itself.

Images exhibit

* inconsistent framing
* varying face scales
* different body visibility
* cluttered backgrounds
* large appearance variation

These characteristics make learning robust facial representations from scratch difficult.

---

## Final Decision

The experiments strongly justify moving to transfer learning.

Rather than continuing to increase CNN complexity, pretrained ImageNet feature extractors are expected to provide substantially richer low-level and mid-level visual representations while requiring fewer training examples.

---

## Lessons Learned

During this project I initially assumed model architecture was the primary cause of poor performance. Through controlled experimentation I learned that reproducible ML development depends equally on deterministic pipelines, careful experiment tracking, and understanding dataset characteristics. This project reinforced that unsuccessful experiments can be just as informative as successful ones when they are systematically designed and documented.

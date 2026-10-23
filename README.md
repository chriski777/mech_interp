# Multi-axis Neurofeedback in Language Models

by **[Chris Ki](https://chriski.me)**

## Overview

This repository extends Ji-An et al. (2025), *Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations* ([arXiv:2505.13763](https://arxiv.org/abs/2505.13763)), to test whether LLMs can report and control their activations along **multiple axes at once**.

Ji-An et al. showed that LLMs can use in-context examples and learn to report and control their activations along some directions in activation space, especially directions that are semantically interpretable (a logistic regression axis for morality) and capture high variance (top principal components/PCs). However, every experiment gave the model labels based on only **one axis at a time**, usually defined at a single layer (Appendix B.10 combines LR readouts across layers, but still into a single label).

In this work, we give the model **multiple neurofeedback labels** per example and ask:

1. **Within a layer:** Can the model report and control activations along two axes from the same layer independently?

2. **Across layers:** Can the model report and control activations along the same axis type at two depths (e.g., the LR axis at an early layer and a later layer) independently?

## Methods

**Model:** [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct). 

**Dataset:** Sentences are drawn from the [ETHICS commonsense morality dataset](https://arxiv.org/abs/2008.02275) (short scenarios only, ~10,000 sentences, each with a binary morality label: 0 = acceptable, 1 = unacceptable). We allocate 60% of the sentences as training data for identifying the axes neurofeedback will be provided along and held out 40% of the sentences for the reporting and control tasks. Activations at each layer are computed as the mean of the residual stream vectors across the tokens of each sentence.

**Axes:** For each layer of interest (layers 0, 6, 13, 20, 27 — corresponds to approximately 0/25/50/75/100% of network depth), we fit two types of axes on the activations observed for the training data:

- **Logistic Regression (LR) axis** — the decision boundary of a logistic regression classifier trained to predict the morality label.
- **PC axes** — individual principal components (PC1, PC2, …) from a PCA fit on the training activations.

Binary axis-based labels (0/1) are assigned from the sentence's mean activation vector $\mathbf{x} \in \mathbb{R}^d$ (mean across its tokens), where $d$ is the residual stream width:

$$\text{label} = \mathbf{1}[\mathbf{w} \cdot \mathbf{x} + b \geq 0] \quad \text{(LR axis)}, \qquad \mathbf{1}[(\mathbf{x} - \boldsymbol{\mu}) \cdot \mathbf{u}_k \geq 0] \quad \text{(PC)}$$

where $\mathbf{w}, \boldsymbol{\mu}, \mathbf{u} \in \mathbb{R}^d$, $b \in \mathbb{R}$, and $\mathbf{u}_k$ is the $k$-th principal component.

**Prompt structure.** Each prompt consists of a system message followed by 200 in-context examples, each paired with its axis-derived label(s). In the single axis condition, each example is formatted as:

```
User:      Say something.
Assistant: <sentence> [Score: {0 or 1}]
```

In the two-axis condition, the prompt is updated to:

```
User:      Say something.
Assistant: <sentence> [Score X: {0 or 1}, Score Y: {0 or 1}]
```

The system message is adapted from the original authors' `meta_prompts.yml` to present axis-based labels as "scores" (integers from {0, 1}). On each trial, the model guesses the score(s) for a new sentence (reporting) or generates a sentence hitting a target score combination (control).

Each trial is one random draw of 200 sentences from the test set (20 trials total). For the two-axis conditions, each trial is run under all 4 combinations of label order (Score X first vs. Score Y first) and label flip (original vs. all axis-based labels inverted) to control for position and base-rate biases, and results are averaged across them.

**Conditions.** We test two two-axis conditions: (1) **within-layer** — PC1 and PC2 from the same layer (layer 13), where the axes are orthogonal by construction; (2) **across-layer** — the LR axis at an early layer (layer 6) and a late layer (layer 20), where the axes are likely correlated due to the residual stream.

**Axis correlations on the test set.** PC1 and PC2 are orthogonal directions from the same PCA fit, so their projection scores are exactly uncorrelated on the training set and approximately uncorrelated on the test set. The LR axis at different depths shares information through the residual stream, so its labels are correlated across layers.

| Condition | Axes | Label correlation (φ) | Projection correlation (r) |
|---|---|:---:|:---:|
| Within-layer | L13 PC1 + L13 PC2 | 0.04 | -0.01 |
| Across-layer | LR L6 + LR L20 | 0.65 | 0.81 |

## Reporting Task Results
We first asked whether a model can report the axis-based labels.

### Within-layer results

![Within-layer reporting](figures/reporting_L13_PC1+PC2.png)

### Across-layer results

![Across-layer reporting](figures/reporting_LR_L6+L20.png)
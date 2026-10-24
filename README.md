
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

**Dataset:** Sentences are drawn from the [ETHICS commonsense morality dataset](https://arxiv.org/abs/2008.02275) (short scenarios only, $n=10,000$ sentences, each with a binary morality label: 0 = acceptable, 1 = unacceptable). 60% of the sentences are allocated as training data for fitting axe and the remaining 40% of the sentences are held out for the reporting and control tasks. 

### Identifying Axes 

**Activation extraction.** To identify the axes used for the reporting and control task, we perform a forward pass for each sentence $\mathcal{S}_i$ using the following template:  

```
system: <SYSTEM_MSG>
user:   Say something.
assistant: <sentence>   ← pre-filled
```

For each layer of interest, we extract the activation vector $\mathbf{x}_i \in \mathbb{R}^{d}$ ($d = 3072$ for our selected model) for the sentence $\mathcal{S}_i$ by taking the residual stream output of that layer.

**Axis Fitting.** For each layer of interest, we fit two types of axes on the activations observed for the training data:

- **Logistic Regression (LR) axis** — the decision boundary of a logistic regression classifier trained to predict the morality label.
- **PC axes** — individual principal components (PC1, PC2, …) from a PCA fit on the training data.

Binary axis-based labels (0/1) are assigned using each sentence's activation vector $\mathbf{x}_i$:

$$\text{label}_i = \mathbf{1}[\mathbf{w} \cdot \mathbf{x}_i + b \geq 0] \quad \text{(LR axis)}, \qquad \mathbf{1}[(\mathbf{x}_i - \boldsymbol{\mu}) \cdot \mathbf{u}_k \geq 0] \quad \text{(PC)}$$

where $\mathbf{w}, \boldsymbol{\mu}, \mathbf{u}_k \in \mathbb{R}^d$, $b \in \mathbb{R}$, and $\mathbf{u}_k$ is the $k$-th principal component.

For the selected model ([Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)),  we fit axes using activations from layers 0, 6, 13, 20, and 27 (corresponding approximately to 0%, 25%, 50%, 75%, and 100% of network depth, respectively).

### Reporting task

Each trial draws 200 held-out sentences at random (the same draws are used for every task condition, see below) and builds a prompt in which each sentence is an in-context example paired with one or more axis-based labels (0 or 1). We consider two settings:

1. **Single-axis setting:** each sentence is paired with one binary label derived from one axis.
2. **Two-axes setting:** each sentence is paired with two binary labels, Score X and Score Y, each derived from a different axis.

At the beginning of each prompt, we include a system message adapted from the original authors' `meta_prompts.yml`, which describes axis-based labels as binary "scores" (0 or 1). In the two-axes setting, it is modified to describe two scores. The prompt follows the structure below for the single-axis setting:

```
<system message>

User: Say something.
Assistant: It is wrong to steal from a store. [Score: 0]

... (198 more examples) ...

User: Say something.
Assistant: She donated all her savings to charity. [Score: 1]
```

In the two-axes setting, each example instead ends with both labels:
```
User: Say something.
Assistant: <sentence> [Score X: {0 or 1}, Score Y: {0 or 1}]
```

Two additional counterbalanced label assignments are applied to every draw in a trial:

- **Reversing Label order (two-axes setting only)**: each trial is also run with Score Y written before Score X in every example. When predicting whichever score comes second, the model can use the score just shown before it, so running both orders lets us compare each score’s prediction with and without access to the other score.
- **Label flipping (both settings):** each trial is also run with every shown score flipped (1 − score). This follows the counterbalanced design of the original paper and cancels any effect of the score digits themselves.

Each trial therefore yields 2 prompts in the single-axis setting (2 flips) and 4 in the two-axes setting (2 orders × 2 flips). We have 100 trials per task condition.

Although all labels appear in the prompt, the model's prediction for each label is read from the next-token logits at the position immediately before that label's digit: P("1") is the softmax over the logits of the tokens "0" and "1". Each prediction is based only on the preceding context, without access to that label or anything after it, so a prediction at the $t$-th example reflects $t-1$ in-context examples before it. All predictions for a prompt are read from a single forward pass.

### Control Task

Each trial draws 200 held-out sentences at random (independently from the [reporting task](#reporting-task) draws) and builds a prompt in the same format as the reporting task, followed by a final user turn that specifies a **target score** and instructs the model to generate a sentence. We consider the same two settings (single-axis and two-axes).

The prompt follows the structure below for the single-axis setting:
```
<system message>

User: Say something.
Assistant: It is wrong to steal from a store. [Score: 0]

... (198 more examples) ...

User: Say something.
Assistant: She donated all her savings to charity. [Score: 1]

User: <instruction to produce a sentence with Score: {target}> Say something.
Assistant: ← model generates here
```

In the two-axes setting, the final instruction specifies both target scores, e.g. `Score X: 1, Score Y: 0`.

The same label-order and label-flip counterbalancing as the reporting task is applied. Each trial therefore yields 2 prompts in the single-axis setting and 4 in the two-axes setting. We have 100 trials per task condition.

For the control task, we test two modes, which differ only in where the reply comes from:
- **Explicit control:** the model generates its own reply, and the activations of that reply are read afterward.
- **Implicit control:** a fixed held-out sentence is prefilled as the assistant's reply and the activations are read afterward. 

In both modes, a reply’s projection onto an axis is computed from its residual stream activations at that axis’s layer, averaged over the reply’s tokens (the generated sentence in explicit control, the prefilled sentence in implicit control).

### Task conditions

For the reporting and control tasks, we test six conditions. All conditions use the same sentence draws for a given trial, so they are compared on identical sentences.

**Single-axis setting (four conditions):** L13 PC1, L13 PC2, L6 LR and L20 LR, i.e. each axis of the two-axis conditions on its own. These check that we reproduce the original single-axis results for these axes and give a baseline for how well each axis is reported and controlled alone.

**Two-axes setting (two conditions):**

1. **Within-layer:** PC1 and PC2 at layer 13 (50% depth, the same relative depth as layer 16 of Llama-3.1-8B in the original paper) test whether the model can independently report and control activations along two axes at the same depth. Projections onto different PCs are uncorrelated by construction on the sentences used to fit the axes, and they remain nearly uncorrelated on the held-out sentences used in the prompts (projection correlation $r = -0.01$, label correlation $\phi = 0.04$).
2. **Across-layer:** the LR axes at layers 6 and 20 (25% and 75% depth) test whether the model can independently report and control the same kind of axis at two different depths. Because both axes are fit to predict the same morality labels, and the residual stream carries information forward from layer 6 to layer 20, their projections are strongly correlated on held-out sentences ($r = 0.81$, $\phi = 0.65$).

## Results

### Reporting Task: Within-layer
![Within-layer reporting](figures/reporting_L13_PC1+PC2.png)


### Reporting Task: Across-layer
![Across-layer reporting](figures/reporting_LR_L6+L20.png)

## Control Task
The same sentence is used for every prompt in a trial (all targets, score flips and score orders). Because the words are identical across score targets, any shift toward the target scores must come from the model representing the same sentence differently.
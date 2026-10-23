# Multi-axis Neurofeedback in Language Models

by **[Chris Ki](https://chriski.me)**

This repo extends Ji-An et al. (2025), *Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations* ([arXiv:2505.13763](https://arxiv.org/abs/2505.13763)), to test whether LLMs can report and control their activations along **multiple axes at once**.

Ji-An et al. showed that LLMs can use in-context examples and learn to report and control their activations along some directions in activation space, especially directions that are semantically interpretable (a logistic regression axis for morality) and capture high variance (top principal components/PCs). However, every experiment gave the model labels based on only **one axis at a time**, usually defined at a single layer (Appendix B.10 combines LR readouts across layers, but still into a single label).

In this work, we give the model **multiple neurofeedback labels** per example and ask:

1. **Within a layer:** Can the model report and control activations along two axes from the same layer independently?

2. **Across layers:** Can the model report and control activations along the same axis type at two depths (e.g., the LR axis at an early layer and a later layer) independently?
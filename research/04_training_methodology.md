## 4. Training Methodology

### 4.1 Multi-Phase Training Curriculum

We propose a curriculum-based training approach with four progressive phases:

1. **Supervised Learning Phase**: Train the model to recognize knowledge boundaries using labeled examples.
2. **Bridge Detection Phase**: Focus on accurate activation of bridge neurons at appropriate knowledge boundaries.
3. **Bridge Retrieval Phase**: Train the model to generate effective neural query representations.
4. **Integration Phase**: Optimize the model's ability to incorporate retrieved information into its generation process.

This phased approach allows the model to progressively learn the complex task of knowledge integration.

### 4.2 Loss Functions

The training objective combines multiple loss terms:

1. **Language Modeling Loss**: Standard next-token prediction loss.
2. **Bridge Detection Loss**: Binary classification loss for knowledge boundary detection.
3. **Query Quality Loss**: Measures the quality of generated queries against ground truth.
4. **Integration Loss**: Measures the quality of the generated text with retrieval compared to expert demonstrations.

The combined loss function is:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{LM} + \lambda_2 \mathcal{L}_{BD} + \lambda_3 \mathcal{L}_{QQ} + \lambda_4 \mathcal{L}_{INT}$$

where $\lambda_1, \lambda_2, \lambda_3, \lambda_4$ are weighting hyperparameters that can be adjusted according to the training phase.

### 4.3 Parameter-Efficient Fine-Tuning

To efficiently adapt pre-trained language models, we employ parameter-efficient fine-tuning techniques. Only the bridge detector, query encoder, and response integrator parameters are fully trainable, while the base model receives limited updates through LoRA adapters.

Specifically, for transformer weights $W$, we add low-rank updates:

$$W' = W + \Delta W = W + A \cdot B$$

where $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times d}$ with rank $r \ll d$. This approach significantly reduces the number of trainable parameters while allowing effective adaptation.
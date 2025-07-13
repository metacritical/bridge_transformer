## 2. Related Work

### 2.1 Retrieval-Augmented Language Models

Retrieval-augmented language models enhance generation capabilities by incorporating external knowledge. REALM [4] and RAG [5] pioneered this approach, using dense retrievers to fetch relevant documents that are then provided as additional context for language modeling. Subsequent work has refined these approaches with improved retrievers [6], rerankers [7], and more sophisticated integration methods [8]. Recent advances include RETRO [22], which retrieves from trillions of tokens, Self-RAG [27] for self-reflection, and corrective RAG [29] for improved accuracy. Comprehensive surveys [16, 23, 30] highlight the rapid evolution of this field.

### 2.2 Parameter-Efficient Fine-Tuning

Our approach draws inspiration from parameter-efficient fine-tuning methods, particularly LoRA (Low-Rank Adaptation) [9] and adapter-based approaches [10]. These methods modify a small subset of parameters while keeping most of the model frozen, enabling efficient adaptation to new tasks. We extend this concept by repurposing specific neurons for knowledge boundary detection.

### 2.3 Neural Module Networks and Mixture-of-Experts

Neural module networks [11] and mixture-of-experts architectures [12] employ specialized neural components for different aspects of a task. Similarly, our approach uses dedicated bridge neurons for knowledge boundary detection and integration, but differs in how these components are integrated into the base architecture.

### 2.4 External Memory Mechanisms

External memory architectures such as Neural Turing Machines [13] and Memory Networks [14] augment neural networks with explicit memory components. Recent work has shown that transformer feed-forward layers naturally function as key-value memories [24], and persistent memory mechanisms can augment self-attention [21]. Extensions like Transformer-XL [25] and ALiBi [26] address context length limitations. Our approach shares the goal of expanding the model's knowledge capacity but focuses on creating direct neural pathways to external knowledge sources rather than training end-to-end differentiable memory systems.

### 2.5 Neural Pruning and Sparsity

Our Pruning-Guided Bridge Allocation (PGBA) approach builds on extensive research in neural network pruning. The lottery ticket hypothesis [17] demonstrates that sparse subnetworks can match dense network performance. Classical work on optimal brain damage [18] and modern approaches using L0 regularization [19] provide theoretical foundations for identifying less critical neurons. These methods inform our approach to repurposing neurons with minimal impact on model performance.

### 2.6 Adaptive Computation

Recent work on adaptive computation in language models is relevant to our bridge activation mechanism. Confident Adaptive Language Modeling (CALM) [20] demonstrates that models can learn when to stop computation early, similar to how our bridge neurons learn when external knowledge is needed. This adaptive approach ensures efficient use of computational resources.

### 2.7 Biological Inspiration

Our bridge mechanism draws inspiration from biological systems where specialized neural pathways connect different functional areas of the brain [15]. Gateway neurons in cognitive systems serve as interfaces between different processing modules, similar to how our bridge neurons facilitate communication between the language model and external knowledge sources.
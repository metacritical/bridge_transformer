#!/bin/bash

# Script to fix website image issues

# Check if Jekyll is installed
if ! command -v jekyll &> /dev/null
then
    echo "Jekyll is not installed. Please install it before proceeding."
    exit 1
fi

# Ensure the images directory exists
mkdir -p website/assets/images

# Copy all SVG files to the images directory
cp figures/*.svg website/assets/images/

# Fix the Jekyll config
cat > website/_config.yml << 'EOL'
title: Bridge Neural Networks
description: Direct Neural Pathways for External Knowledge Integration
author: Pankaj Doharey

# Social media links
github_username: pankajdoharey
email: pankajdoharey@zendiffusion.art
website: https://zendiffusion.art

# Site configuration
theme: minima
show_excerpts: true
permalink: /:title/

# Asset settings
baseurl: ""
url: ""

# Configure image paths
defaults:
  - scope:
      path: "assets/images"
    values:
      image: true

header_pages:
  - paper.md
  - about.md
EOL

# Create paper.md with correct image paths
cat > website/paper.md << 'EOL'
---
layout: page
title: Bridge Neural Networks
description: Direct Neural Pathways for External Knowledge Integration
---

# Bridge Neural Networks: Direct Neural Pathways for External Knowledge Integration

**Pankaj Doharey**  
ZenDiffusion.art  
pankajdoharey@zendiffusion.art

## Abstract

Large language models (LLMs) face inherent limitations in knowledge access and factuality, constrained by their parametric knowledge representations. While retrieval-augmented generation (RAG) has emerged as a solution, it suffers from context window pollution, reduced reasoning capacity, and unnatural integration of external information. We propose Bridge Neural Networks (BNNs), a novel architecture that repurposes a subset of neurons to create dedicated neural pathways for external knowledge access. Unlike RAG, BNNs detect knowledge boundaries through trained neuron activations, generate neural query representations, and integrate external information directly at the hidden state level without consuming context tokens. We present a theoretical foundation for BNNs, detail their architecture, outline training methodology, and propose evaluation frameworks that measure factuality, reasoning preservation, and integration quality. Our analysis suggests BNNs offer a more elegant and efficient approach to knowledge integration that preserves model reasoning capacity while enabling selective external information access.

**Keywords**: neural networks, language models, knowledge integration, external memory, retrieval

## 1. Introduction

Large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation. However, they face fundamental limitations in knowledge access and factuality, as they rely on parametric knowledge encoded in their weights during training. This creates several challenges:

1. **Knowledge Limitations**: Even the largest models cannot encode all potentially useful information.
2. **Knowledge Staleness**: Models trained on static corpora cannot access information that emerges after training.
3. **Hallucination**: Models often generate plausible but factually incorrect information when operating beyond their knowledge boundaries.

The dominant approach to address these limitations has been retrieval-augmented generation (RAG), which retrieves relevant documents from external sources and injects them into the context window. While effective, RAG introduces significant drawbacks:

1. **Context Window Pollution**: Retrieved documents consume precious context tokens, reducing the space available for user queries and reasoning.
2. **Integration Artifacts**: The separation between retrieval and generation creates artificial boundaries in the generation process.
3. **Inefficient Retrieval**: Retrieval occurs regardless of whether it's necessary, often wasting computational resources.
4. **Prompting Complexity**: Complex prompt engineering is required to format retrieved information effectively.

We propose Bridge Neural Networks (BNNs), a novel architecture that addresses these limitations by creating direct neural pathways for external knowledge access. Rather than injecting retrieved information into the context window, BNNs repurpose a small subset of neurons to detect knowledge boundaries, generate neural query representations, and integrate external information directly at the hidden state level.

## Key Figures

### Bridge Neural Network Architecture
![Bridge Neural Network Architecture](/assets/images/figure1_bridge_architecture.svg)

*Figure 1: Bridge Neural Network Architecture showing the base transformer model with bridge neurons and external knowledge service connections.*

### Information Flow Comparison
![Information Flow Comparison](/assets/images/figure2_information_flow_comparison.svg)

*Figure 2: Comparison of information flow in traditional RAG (left) versus Bridge Neural Networks (right).*

### Pruning-Guided Bridge Allocation
![Pruning-Guided Bridge Allocation](/assets/images/figure3_pruning_bridge_allocation.svg)

*Figure 3: Pruning-Guided Bridge Allocation process.*

### Mathematical Framework
![Mathematical Framework](/assets/images/figure4_mathematical_framework.svg)

*Figure 4: Mathematical Framework for Bridge Neural Networks.*

### Training Curriculum
![Training Curriculum](/assets/images/figure5_training_curriculum.svg)

*Figure 5: Bridge Neural Network Training Curriculum showing the four phases of training.*

## Conclusion

Bridge Neural Networks represent a novel approach to integrating external knowledge into language models without the limitations of traditional retrieval-augmented generation. By repurposing a small subset of neurons to create dedicated neural pathways for knowledge access, BNNs maintain the model's reasoning capabilities while enabling selective and efficient access to vast external knowledge sources.

[Read the full paper](bridge_neural_networks.pdf)
EOL

echo "✅ Website fixed. Run the following to start the website:"
echo "cd website && bundle exec jekyll serve"
echo "Then open http://localhost:4000 in your browser."

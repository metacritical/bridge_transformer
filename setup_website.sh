#!/bin/bash

# Setup script for academic website using Jekyll
# This script sets up a Jekyll website for the Bridge Neural Networks paper

# Check if git is installed
if ! command -v git &> /dev/null
then
    echo "Git is not installed. Please install it before proceeding."
    exit 1
fi

# Check if Ruby and Jekyll are installed
if ! command -v jekyll &> /dev/null
then
    echo "Jekyll is not installed. Please install it before proceeding."
    echo "You can install Jekyll by following the instructions at: https://jekyllrb.com/docs/installation/"
    exit 1
fi

# Create the website directory if it doesn't exist
mkdir -p website

# Clone the research paper template if it doesn't exist
if [ ! -d "website/_site" ]; then
    echo "Cloning the research paper template..."
    git clone https://github.com/metacritical/research-paper-template.git website/
    
    # Remove git repository
    rm -rf website/.git
fi

# Copy figures to website assets
mkdir -p website/assets/images
cp figures/*.svg website/assets/images/

# Create the content from markdown
echo "Creating website content from markdown..."

# Create index.md (homepage)
cat > website/index.md << 'EOL'
---
layout: home
title: Bridge Neural Networks
description: Direct Neural Pathways for External Knowledge Integration
---

# Bridge Neural Networks

## Direct Neural Pathways for External Knowledge Integration

**Pankaj Doharey**  
ZenDiffusion.art  
pankajdoharey@zendiffusion.art

### Abstract

Large language models (LLMs) face inherent limitations in knowledge access and factuality, constrained by their parametric knowledge representations. While retrieval-augmented generation (RAG) has emerged as a solution, it suffers from context window pollution, reduced reasoning capacity, and unnatural integration of external information. 

We propose Bridge Neural Networks (BNNs), a novel architecture that repurposes a subset of neurons to create dedicated neural pathways for external knowledge access. Unlike RAG, BNNs detect knowledge boundaries through trained neuron activations, generate neural query representations, and integrate external information directly at the hidden state level without consuming context tokens. 

We present a theoretical foundation for BNNs, detail their architecture (which we refer to as the "Abhay Architecture"), outline training methodology, and propose evaluation frameworks that measure factuality, reasoning preservation, and integration quality. Our analysis suggests BNNs offer a more elegant and efficient approach to knowledge integration that preserves model reasoning capacity while enabling selective external information access.

[Read the full paper](paper.html)
EOL

# Convert the markdown document to a Jekyll page
cat > website/paper.md << 'EOL'
---
layout: page
title: Bridge Neural Networks
description: Direct Neural Pathways for External Knowledge Integration
---
EOL

# Append the content of the markdown document with corrected image paths
cat markdown/bridge_neural_networks.md | sed 's/figures\//\/assets\/images\//g' >> website/paper.md

# Create an about page
cat > website/about.md << 'EOL'
---
layout: page
title: About
description: About the Bridge Neural Networks project
---

# About the Project

The Bridge Neural Networks project introduces a novel architecture ("Abhay Architecture") for integrating external knowledge into language models without context window pollution.

## Author

**Pankaj Doharey**  
Researcher at ZenDiffusion.art  
Contact: pankajdoharey@zendiffusion.art

## Research Focus

This research focuses on creating direct neural pathways for external knowledge access in language models. The Abhay Architecture allows models to:

1. Detect knowledge boundaries through trained neuron activations
2. Generate neural query representations
3. Integrate external information directly at the hidden state level

This approach preserves the model's reasoning capacity while enabling selective access to vast external knowledge sources.

## Publications

- **Bridge Neural Networks: Direct Neural Pathways for External Knowledge Integration** (April 2025)

## Related Work

This research builds upon work in:
- Retrieval-augmented language models
- Parameter-efficient fine-tuning
- Neural module networks
- External memory mechanisms
- Biological neural pathways
EOL

# Create a configuration file
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

header_pages:
  - paper.md
  - about.md
EOL

# Create a Gemfile if it doesn't exist
if [ ! -f "website/Gemfile" ]; then
    echo "Creating Gemfile..."
    cat > website/Gemfile << 'EOL'
source "https://rubygems.org"

gem "jekyll", "~> 4.2"
gem "webrick", "~> 1.7"  # Needed for Ruby 3.0+
gem "minima", "~> 2.5"   # The minima theme

# If you want to use GitHub Pages
# gem "github-pages", group: :jekyll_plugins

# Jekyll plugins
group :jekyll_plugins do
  gem "jekyll-feed"
  gem "jekyll-seo-tag"
  gem "jekyll-sitemap"
end
EOL
    echo "Gemfile created."
fi

echo "✅ Website setup completed."
echo "To test the website locally, run:"
echo "cd website && bundle exec jekyll serve"
echo "Then open http://localhost:4000 in your browser."
echo ""
echo "To build the website for deployment, run:"
echo "cd website && bundle exec jekyll build"
echo "The output will be in website/_site/"

#!/bin/bash

# Setup website for Bridge Neural Networks paper
# This script builds the Jekyll website with your actual paper content

echo "Setting up Bridge Neural Networks website..."

# Navigate to website directory
cd website

# Load Ruby environment
source ~/.bash_profile

# Build Jekyll site
echo "Building Jekyll site with Bridge Neural Networks content..."
bundle exec jekyll build

if [ $? -eq 0 ]; then
    echo "✅ Website built successfully!"
    echo "📄 Website available at: website/_site/index.html"
    echo "🎨 Beautiful academic template with your actual Bridge Neural Networks content"
    echo "🔗 Professional design showing your research instead of Lorem ipsum"
    
    # Start local server for testing
    echo "Starting local server for testing..."
    echo "🌐 Website will be available at: http://localhost:4000"
    echo "Press Ctrl+C to stop the server"
    
    bundle exec jekyll serve --port 4000 --host 0.0.0.0
else
    echo "❌ Website build failed"
    exit 1
fi

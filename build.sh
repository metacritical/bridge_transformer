#!/bin/bash

# Bridge Neural Networks Build Script
# This script can generate different formats of the research paper:
# 1. PDF from Markdown using pandoc
# 2. Academic website using Jekyll
# 3. Academic PDF using LaTeX

# Function to convert SVG files to PNG
convert_svg_to_png() {
  echo "Converting SVG files to PNG..."
  chmod +x convert_to_png.sh
  ./convert_to_png.sh
}

# Function to display usage information
show_usage() {
  echo "Usage: $0 [option] [subcommand]"
  echo "Options:"
  echo "  markdown         Generate PDF from Markdown using pandoc"
  echo "  website          Build academic website (default behavior)"
  echo "  website build    Build Jekyll website only"
  echo "  website serve    Serve Jekyll website (builds first if needed)"
  echo "  latex            Generate academic PDF using the LaTeX template"
  echo "  all              Generate all formats"
  echo "  help             Show this help message"
  echo ""
  echo "Examples:"
  echo "  $0 website build    # Build website to _site directory"
  echo "  $0 website serve    # Start local server at http://localhost:4000"
}

# Function to check if a command is available
check_command() {
  if ! command -v $1 &> /dev/null
  then
    echo "$1 is not installed. Please install it before proceeding."
    return 1
  fi
  return 0
}

# Generate PDF from Markdown
generate_markdown_pdf() {
  echo "Generating PDF from Markdown..."
  
  # Check if pandoc is installed
  if ! check_command pandoc; then
    echo "Please install Pandoc: https://pandoc.org/installing.html"
    return 1
  fi
  
  # Create output directory if it doesn't exist
  mkdir -p output
  
  # Copy figures to output directory so relative paths work in HTML
  cp -r figures output/
  
  # Create custom template with email field
  cat > /tmp/email_template.html << 'EOF'
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" lang="" xml:lang="">
<head>
  <meta charset="utf-8" />
  <meta name="generator" content="pandoc" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes" />
  <meta name="author" content="$author$" />
  <title>$title$</title>
  <style>
    body { font-family: Times, serif; margin: 1in; line-height: 1.6; }
    h1,h2,h3 { color: #333; }
    .title-section { text-align: center; margin-bottom: 2em; border-bottom: 1px solid #ccc; padding-bottom: 1em; }
    .author-info { margin: 0.5em 0; }
    .email { font-style: italic; color: #666; }
  </style>
</head>
<body>
<div class="title-section">
  <h1>$title$</h1>
  <div class="author-info"><strong>$author$</strong></div>
  <div class="email" style="font-size: 14pt; margin: 8px 0;"><strong>$email$</strong></div>
  <div class="author-info">$institute$</div>
  <div class="author-info">$date$</div>
</div>
$body$
</body>
</html>
EOF

  # Generate HTML with custom template
  pandoc markdown/bridge_neural_networks.md \
    --from markdown \
    --to html \
    --output output/bridge_neural_networks.html \
    --template /tmp/email_template.html \
    --metadata title="Bridge Neural Networks for External Knowledge Integration" \
    --metadata author="Pankaj Doharey" \
    --metadata date="April 2025" \
    --metadata institute="ZenDiffusion.art" \
    --metadata email="pankaj@zendiffusion.art"
  
  # Create LaTeX template with proper packages for pandoc compatibility
  cat > /tmp/pdf_template.latex << 'EOF'
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{times}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{enumitem}
\usepackage{url}
\usepackage{hyperref}

% Define tightlist for pandoc compatibility
\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}

\title{$title$}
\author{$author$\\$email$\\$institute$}
\date{$date$}
\begin{document}
\maketitle
$body$
\end{document}
EOF

  # Add LaTeX to PATH if it exists
  if [ -d "/usr/local/texlive/2024/bin/universal-darwin" ]; then
    export PATH="/usr/local/texlive/2024/bin/universal-darwin:$PATH"
  fi
  
  # Generate PDF directly from markdown using pandoc with default template
  echo "Attempting PDF generation with LaTeX..."
  pandoc markdown/bridge_neural_networks.md \
    --from markdown \
    --to pdf \
    --output output/bridge_neural_networks_md.pdf \
    --metadata title="Bridge Neural Networks for External Knowledge Integration" \
    --metadata author="Pankaj Doharey" \
    --metadata subtitle="pankaj@zendiffusion.art" \
    --metadata date="April 2025" \
    --metadata institute="ZenDiffusion.art"
  
  # Check if PDF was actually created
  if [ -f "output/bridge_neural_networks_md.pdf" ]; then
    echo "✅ Markdown PDF generated successfully: output/bridge_neural_networks_md.pdf"
  else
    echo "❌ PDF generation failed - LaTeX not found or conversion error"
    echo "📄 HTML version available: output/bridge_neural_networks.html"
  fi
}

# Prepare website assets (figures and dependencies)
prepare_website_assets() {
  # Ensure PNG files exist (convert if needed)
  if [ ! -d "figures/png" ] || [ -z "$(ls -A figures/png 2>/dev/null)" ]; then
    echo "📷 Converting SVG figures to PNG for website..."
    convert_svg_to_png
  fi
  
  # Copy PNG figures to website assets
  echo "📁 Copying figures to website assets..."
  mkdir -p website/assets/images
  cp figures/png/*.png website/assets/images/ 2>/dev/null || echo "⚠️  No PNG files to copy"
}

# Build the academic website (no serving)
build_website() {
  echo "Building academic website..."
  
  # Prepare assets
  prepare_website_assets
  
  # Navigate to website directory
  cd website || { echo "❌ Website directory not found"; return 1; }
  
  # Load Ruby environment
  source ~/.bash_profile
  
  # Build Jekyll site only
  echo "Building Jekyll site with Bridge Neural Networks content..."
  bundle exec jekyll build
  
  if [ $? -eq 0 ]; then
    echo "✅ Website built successfully!"
    echo "📄 Website available at: website/_site/index.html"
    echo "🎨 Beautiful academic template with your actual Bridge Neural Networks content"
    cd ..
  else
    echo "❌ Website build failed"
    cd ..
    return 1
  fi
}

# Serve the academic website
serve_website() {
  echo "Serving academic website..."
  
  # Prepare assets first
  prepare_website_assets
  
  # Navigate to website directory
  cd website || { echo "❌ Website directory not found"; return 1; }
  
  # Load Ruby environment
  source ~/.bash_profile
  
  # Check if site is already built
  if [ ! -d "_site" ]; then
    echo "📦 Site not built yet, building first..."
    bundle exec jekyll build
    if [ $? -ne 0 ]; then
      echo "❌ Build failed"
      cd ..
      return 1
    fi
  fi
  
  # Start local server
  echo "Starting local server..."
  echo "🌐 Website will be available at: http://localhost:4000"
  echo "Press Ctrl+C to stop the server"
  
  bundle exec jekyll serve --port 4000 --host 0.0.0.0
}

# Generate LaTeX PDF
generate_latex_pdf() {
  echo "Generating LaTeX PDF..."
  
  # Create tau-class directory if it doesn't exist
  mkdir -p latex/tau-class
  
  # Check if LaTeX files exist, if not create them
  if [ ! -f "latex/tau-class/tau.cls" ]; then
    echo "Creating LaTeX template files..."
    # Extract template files or create them
    ./setup_latex_template.sh
  fi
  
  # Convert SVG figures to PDF for LaTeX
  chmod +x convert_figures.sh
  ./convert_figures.sh
  
  # Check if pdflatex is installed
  if ! check_command pdflatex; then
    echo "Please install LaTeX: https://www.latex-project.org/get/"
    return 1
  fi
  
  # Create output directory if it doesn't exist
  mkdir -p output
  
  # Navigate to LaTeX directory
  cd latex
  
  # Generate PDF
  pdflatex bridge_paper.tex
  bibtex bridge_paper
  pdflatex bridge_paper.tex
  pdflatex bridge_paper.tex
  
  # Check if PDF was created
  if [ -f "bridge_paper.pdf" ]; then
    # Move PDF to output directory
    mv bridge_paper.pdf ../output/bridge_paper.pdf
    echo "✅ LaTeX PDF generated successfully: output/bridge_paper.pdf"
  else
    echo "❌ Failed to generate LaTeX PDF."
  fi
  
  # Return to original directory
  cd ..
}

# Main script execution
if [ $# -eq 0 ]; then
  show_usage
  exit 1
fi

case "$1" in
  markdown)
    # Convert SVG to PNG for markdown builds
    convert_svg_to_png
    generate_markdown_pdf
    ;;
  website)
    # Website builds don't need PNG conversion (uses SVG directly)
    if [ "$2" == "build" ]; then
      build_website
    elif [ "$2" == "serve" ]; then
      serve_website
    else
      # Default behavior for backward compatibility - build only
      build_website
    fi
    ;;
  latex)
    # Convert SVG to PNG for LaTeX builds
    convert_svg_to_png
    generate_latex_pdf
    ;;
  all)
    # Convert once for all builds
    convert_svg_to_png
    generate_markdown_pdf
    build_website
    generate_latex_pdf
    ;;
  help)
    show_usage
    ;;
  *)
    echo "Invalid option: $1"
    show_usage
    exit 1
    ;;
esac

exit 0

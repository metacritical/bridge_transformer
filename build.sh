#!/bin/bash

# Bridge Neural Networks Build Script
# This script can generate different formats of the research paper:
# 1. PDF from Markdown using pandoc
# 2. Academic website using Jekyll
# 3. Academic PDF using LaTeX

# Function to display usage information
show_usage() {
  echo "Usage: $0 [option]"
  echo "Options:"
  echo "  markdown    Generate PDF from Markdown using pandoc"
  echo "  website     Set up academic website using Jekyll template"
  echo "  latex       Generate academic PDF using the LaTeX template"
  echo "  all         Generate all formats"
  echo "  help        Show this help message"
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
  
  # Generate PDF from Markdown with better styling
  pandoc markdown/bridge_neural_networks_png.md \
    --from markdown \
    --to pdf \
    --output output/bridge_neural_networks_md.pdf \
    --pdf-engine=pdflatex \
    --template=default \
    --variable geometry:margin=1in \
    --variable fontsize=11pt \
    --variable fontfamily="times" \
    --variable monofont="courier" \
    --variable urlcolor=blue \
    --variable links-as-notes=true \
    --variable colorlinks=true \
    --variable linkcolor=blue \
    --variable title="Bridge Neural Networks: Direct Neural Pathways for External Knowledge Integration" \
    --variable author="Pankaj Doharey" \
    --variable date="April 2025" \
    --variable institute="ZenDiffusion.art" \
    --variable email="pankajdoharey@zendiffusion.art" \
    --include-in-header=<(echo '\usepackage{titling}\pretitle{\begin{center}\LARGE\bfseries}\posttitle{\end{center}\vspace{2em}}') \
    --include-in-header=<(echo '\usepackage{fancyhdr}\pagestyle{fancy}\fancyfoot[C]{\thepage}\fancyfoot[L]{Pankaj Doharey}\fancyfoot[R]{ZenDiffusion.art}') \
    --variable classoption=onecolumn
  
  if [ $? -eq 0 ]; then
    echo "✅ Markdown PDF generated successfully: output/bridge_neural_networks_md.pdf"
  else
    echo "❌ Failed to generate Markdown PDF."
  fi
}

# Set up the academic website
setup_website() {
  echo "Setting up academic website..."
  
  # Run the website setup script
  chmod +x setup_website.sh
  ./setup_website.sh
  
  if [ $? -eq 0 ]; then
    echo "✅ Website setup completed."
  else
    echo "❌ Failed to setup website."
  fi
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
    generate_markdown_pdf
    ;;
  website)
    setup_website
    ;;
  latex)
    generate_latex_pdf
    ;;
  all)
    generate_markdown_pdf
    setup_website
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

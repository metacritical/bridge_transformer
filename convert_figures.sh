#!/bin/bash

# Convert SVG figures to PDF for LaTeX
# This script uses Inkscape or rsvg-convert to convert SVG files to PDF

# Function to check if a command is available
check_command() {
  if ! command -v $1 &> /dev/null
  then
    echo "$1 is not installed."
    return 1
  fi
  return 0
}

# Create directory for PDF figures if it doesn't exist
mkdir -p latex/figures

# Check if either Inkscape or rsvg-convert is available
if check_command inkscape; then
  CONVERTER="inkscape"
  echo "Using Inkscape to convert SVG to PDF..."
elif check_command rsvg-convert; then
  CONVERTER="rsvg-convert"
  echo "Using rsvg-convert to convert SVG to PDF..."
else
  echo "Neither Inkscape nor rsvg-convert is installed. Please install one of them and try again."
  echo "Inkscape: https://inkscape.org/release/"
  echo "rsvg-convert: Usually available in the librsvg package."
  exit 1
fi

# Convert SVG files to PDF
for svg_file in figures/*.svg; do
  filename=$(basename -- "$svg_file")
  filename_noext="${filename%.*}"
  output_file="latex/figures/${filename_noext}.pdf"
  
  echo "Converting $svg_file to $output_file..."
  
  if [ "$CONVERTER" = "inkscape" ]; then
    inkscape --export-filename="$output_file" "$svg_file"
  else
    rsvg-convert -f pdf -o "$output_file" "$svg_file"
  fi
  
  if [ $? -eq 0 ]; then
    echo "✅ Converted $filename to PDF"
  else
    echo "❌ Failed to convert $filename"
  fi
done

echo "Conversion complete!"

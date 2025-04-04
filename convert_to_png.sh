#!/bin/bash

# Create PNG versions of all SVG files
echo "Converting SVG files to PNG format..."

# Ensure output directories exist
mkdir -p figures/png

# Convert each SVG file to PNG
for svg_file in figures/*.svg; do
  filename=$(basename -- "$svg_file")
  filename_noext="${filename%.*}"
  output_file="figures/png/${filename_noext}.png"
  
  echo "Converting $svg_file to $output_file..."
  
  # Use rsvg-convert for the conversion with high DPI for quality
  rsvg-convert -f png -o "$output_file" -d 300 "$svg_file"
  
  if [ $? -eq 0 ]; then
    echo "✅ Converted $filename to PNG"
  else
    echo "❌ Failed to convert $filename"
  fi
done

echo "All SVG files converted to PNG format."

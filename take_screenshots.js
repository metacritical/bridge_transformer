// Simple script to take screenshots of SVG files using Puppeteer
const fs = require('fs');
const path = require('path');

async function takeScreenshots() {
  // Directory paths
  const figuresDir = path.join(__dirname, 'figures');
  const outputDir = path.join(__dirname, 'output/screenshots');
  
  // Create output directory if it doesn't exist
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  // Get list of SVG files
  const svgFiles = fs.readdirSync(figuresDir)
    .filter(file => file.endsWith('.svg'))
    .map(file => path.join(figuresDir, file));
  
  console.log(`Found ${svgFiles.length} SVG files to screenshot`);
  
  for (const svgFile of svgFiles) {
    const filename = path.basename(svgFile);
    console.log(`Taking screenshot of ${filename}`);
    
    // Output path for the screenshot
    const screenshotPath = path.join(outputDir, `${path.basename(filename, '.svg')}.png`);
    
    // Read SVG content
    const svgContent = fs.readFileSync(svgFile, 'utf8');
    
    // Create a simple HTML file to display the SVG
    const tempHtmlPath = path.join(outputDir, `${path.basename(filename, '.svg')}_temp.html`);
    
    const htmlContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="UTF-8">
        <title>SVG Screenshot: ${filename}</title>
        <style>
          body { margin: 0; padding: 0; background-color: white; }
          .svg-container { display: block; margin: 0 auto; }
        </style>
      </head>
      <body>
        <div class="svg-container">
          ${svgContent}
        </div>
      </body>
      </html>
    `;
    
    fs.writeFileSync(tempHtmlPath, htmlContent);
    console.log(`Created temporary HTML file: ${tempHtmlPath}`);
  }
  
  console.log('HTML files created in output/screenshots. Use Puppeteer to open and screenshot them.');
}

takeScreenshots();

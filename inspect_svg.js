// Script to inspect SVG rendering issues using Puppeteer
const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

async function inspectSVG() {
  try {
    // Create output directory for screenshots
    const screenshotDir = path.join(__dirname, 'output/screenshots');
    if (!fs.existsSync(screenshotDir)) {
      fs.mkdirSync(screenshotDir, { recursive: true });
    }
    
    // Launch browser
    const browser = await puppeteer.launch({
      headless: 'new',
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    
    const page = await browser.newPage();
    
    // Get all SVG files
    const figuresDir = path.join(__dirname, 'figures');
    const svgFiles = fs.readdirSync(figuresDir)
      .filter(file => file.endsWith('.svg'))
      .map(file => path.join(figuresDir, file));
    
    console.log(`Found ${svgFiles.length} SVG files to inspect`);
    
    // Create HTML file to view SVGs
    for (const svgFile of svgFiles) {
      const filename = path.basename(svgFile);
      const htmlContent = `
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="UTF-8">
          <title>SVG Inspection: ${filename}</title>
          <style>
            body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
            .container { width: 100%; overflow: auto; }
            svg { display: block; margin: 0 auto; border: 1px solid #ddd; }
            .controls { margin: 20px; text-align: center; }
            .details { margin: 20px; }
            pre { background: #f5f5f5; padding: 10px; overflow: auto; }
          </style>
        </head>
        <body>
          <h1>SVG Inspection: ${filename}</h1>
          <div class="controls">
            <button id="highlight-text">Highlight Text Elements</button>
            <button id="highlight-overlap">Check Overlap Issues</button>
          </div>
          <div class="container">
            ${fs.readFileSync(svgFile, 'utf8')}
          </div>
          <div class="details">
            <h2>SVG Metadata</h2>
            <pre id="metadata"></pre>
          </div>
          
          <script>
            // Function to highlight all text elements
            document.getElementById('highlight-text').addEventListener('click', () => {
              const texts = document.querySelectorAll('text');
              texts.forEach(text => {
                const originalFill = text.getAttribute('fill') || 'black';
                text.setAttribute('data-original-fill', originalFill);
                text.setAttribute('fill', 'red');
                
                // Add rectangle behind text to show bounding box
                const bbox = text.getBBox();
                const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                rect.setAttribute('x', bbox.x);
                rect.setAttribute('y', bbox.y);
                rect.setAttribute('width', bbox.width);
                rect.setAttribute('height', bbox.height);
                rect.setAttribute('fill', 'rgba(255, 0, 0, 0.2)');
                rect.setAttribute('class', 'highlight-box');
                text.parentNode.insertBefore(rect, text);
                
                console.log('Text:', text.textContent, 'BBox:', bbox);
              });
            });
            
            // Function to check for potential overlap issues
            document.getElementById('highlight-overlap').addEventListener('click', () => {
              // Clean up any previous highlights
              document.querySelectorAll('.highlight-box').forEach(el => el.remove());
              document.querySelectorAll('.overlap-box').forEach(el => el.remove());
              
              const texts = Array.from(document.querySelectorAll('text'));
              const overlaps = [];
              
              // Check each text element against all others
              for (let i = 0; i < texts.length; i++) {
                const t1 = texts[i];
                const b1 = t1.getBBox();
                
                for (let j = i + 1; j < texts.length; j++) {
                  const t2 = texts[j];
                  const b2 = t2.getBBox();
                  
                  // Simple overlap detection
                  if (!(b1.x > b2.x + b2.width || 
                        b1.x + b1.width < b2.x || 
                        b1.y > b2.y + b2.height ||
                        b1.y + b1.height < b2.y)) {
                    
                    overlaps.push([t1, t2]);
                    
                    // Highlight overlapping elements
                    t1.setAttribute('fill', 'red');
                    t2.setAttribute('fill', 'red');
                    
                    // Add rectangles to show overlap
                    const rect1 = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    rect1.setAttribute('x', b1.x);
                    rect1.setAttribute('y', b1.y);
                    rect1.setAttribute('width', b1.width);
                    rect1.setAttribute('height', b1.height);
                    rect1.setAttribute('fill', 'rgba(255, 0, 0, 0.2)');
                    rect1.setAttribute('stroke', 'red');
                    rect1.setAttribute('class', 'overlap-box');
                    
                    const rect2 = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    rect2.setAttribute('x', b2.x);
                    rect2.setAttribute('y', b2.y);
                    rect2.setAttribute('width', b2.width);
                    rect2.setAttribute('height', b2.height);
                    rect2.setAttribute('fill', 'rgba(0, 0, 255, 0.2)');
                    rect2.setAttribute('stroke', 'blue');
                    rect2.setAttribute('class', 'overlap-box');
                    
                    t1.parentNode.insertBefore(rect1, t1);
                    t2.parentNode.insertBefore(rect2, t2);
                  }
                }
              }
              
              // Display results
              console.log('Found ' + overlaps.length + ' potential overlaps');
              overlaps.forEach(pair => {
                console.log('Overlap: "' + pair[0].textContent + '" with "' + pair[1].textContent + '"');
              });
            });
            
            // Extract and display metadata
            const svg = document.querySelector('svg');
            const metadata = {
              viewBox: svg.getAttribute('viewBox'),
              width: svg.getAttribute('width'),
              height: svg.getAttribute('height'),
              textElements: document.querySelectorAll('text').length,
              fontSize: Array.from(document.querySelectorAll('text')).map(t => t.getAttribute('font-size')).filter(Boolean)
            };
            document.getElementById('metadata').textContent = JSON.stringify(metadata, null, 2);
          </script>
        </body>
        </html>
      `;
      
      const htmlFilePath = path.join(screenshotDir, `${path.basename(filename, '.svg')}_inspect.html`);
      fs.writeFileSync(htmlFilePath, htmlContent);
      
      // Open the HTML file
      await page.goto(`file://${htmlFilePath}`);
      
      // Take a screenshot of the default view
      await page.screenshot({ 
        path: path.join(screenshotDir, `${path.basename(filename, '.svg')}_default.png`),
        fullPage: true
      });
      
      // Click the highlight text button
      await page.click('#highlight-text');
      await page.waitForTimeout(500); // Wait for highlights to appear
      
      // Take a screenshot with text highlighted
      await page.screenshot({ 
        path: path.join(screenshotDir, `${path.basename(filename, '.svg')}_text_highlighted.png`),
        fullPage: true
      });
      
      // Click the check overlap button
      await page.click('#highlight-overlap');
      await page.waitForTimeout(500); // Wait for highlights to appear
      
      // Take a screenshot with overlaps highlighted
      await page.screenshot({ 
        path: path.join(screenshotDir, `${path.basename(filename, '.svg')}_overlaps.png`),
        fullPage: true
      });
      
      console.log(`Analyzed ${filename}`);
    }
    
    await browser.close();
    console.log('Analysis complete. Screenshots saved in output/screenshots directory');
    
  } catch (error) {
    console.error('Error inspecting SVGs:', error);
  }
}

// Check if puppeteer is installed
try {
  require.resolve('puppeteer');
  inspectSVG();
} catch (e) {
  console.error('Puppeteer is not installed. Please run: npm install puppeteer');
  console.error('Error:', e.message);
}

const fs = require('fs');
const path = require('path');

console.log("Starting Quant OS automated format script...");

const brainDir = path.join(__dirname, 'brain');
const files = fs.readdirSync(brainDir).filter(f => f.endsWith('.md') && f !== 'Tharun-Kumar-Gajula.md');

let updatedCount = 0;

for (const file of files) {
  const filePath = path.join(brainDir, file);
  let content = fs.readFileSync(filePath, 'utf8');
  let originalContent = content;
  
  // 1. Force the date update
  content = content.replace(/date:\s*2026-03-14/g, 'date: 2026-03-19');
  
  // 2. Inject progress: 0 specifically right after the cluster line if missing anywhere
  if (!/^progress:/m.test(content)) {
    content = content.replace(/^(cluster:\s*.*?)$/m, '$1\nprogress: 0');
  }

  if (content !== originalContent) {
    fs.writeFileSync(filePath, content, 'utf8');
    updatedCount++;
    console.log(`[UPDATED] ${file}`);
  }
}

console.log(`\nSuccessfully formatted ${updatedCount} Markdown nodes.`);

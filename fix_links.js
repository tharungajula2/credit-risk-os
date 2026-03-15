const fs = require('fs');
const path = require('path');

const dir = path.join(process.cwd(), 'brain');
const files = fs.readdirSync(dir).filter(f => f.endsWith('.md'));

let count = 0;
for (const file of files) {
  const p = path.join(dir, file);
  let content = fs.readFileSync(p, 'utf8');
  let original = content;

  // Replace [[Example.md]] with [[Example]]
  content = content.replace(/\.md\]\]/g, ']]');
  
  // Replace [[Example.md|Display]] with [[Example|Display]]
  content = content.replace(/\.md\|/g, '|');

  if (content !== original) {
    fs.writeFileSync(p, content, 'utf8');
    console.log('Fixed:', file);
    count++;
  }
}

console.log('Total files repaired successfully: ', count);

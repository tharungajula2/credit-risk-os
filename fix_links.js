const fs = require('fs');
const path = require('path');
const matter = require('gray-matter');

const brainDir = path.join(__dirname, 'brain');
const files = fs.readdirSync(brainDir).filter(f => f.endsWith('.md'));

for (const file of files) {
  const filePath = path.join(brainDir, file);
  const raw = fs.readFileSync(filePath, 'utf-8');
  
  const parsed = matter(raw);
  const data = parsed.data;
  let content = parsed.content;
  let modified = false;

  // 1. Add progress: 100
  if (data.progress === undefined) {
    data.progress = 100;
    modified = true;
  }

  // 2. Clean links arrays in frontmatter
  if (data.links && Array.isArray(data.links)) {
    const updatedLinks = data.links.map(link => {
      let l = link;
      l = l.replace(/sandbox_1/g, '2_sandbox');
      l = l.replace(/sandbox_2/g, '3_sandbox');
      l = l.replace(/sandbox_3/g, '4_sandbox');
      
      // The user used piped links which are ok, we just replace the base
      l = l.replace(/1_full_pd_model/g, '1_lending_club_credit_risk_masterclass');
      l = l.replace(/2_monitoring_model/g, '1_lending_club_credit_risk_masterclass');
      l = l.replace(/3_lgd_ead_model_rewritten/g, '1_lending_club_credit_risk_masterclass');
      l = l.replace(/4_ecl_cecl_stress_testing_rewritten/g, '1_lending_club_credit_risk_masterclass');
      
      // Cleanup any duplicate full path if it already matches
      if (l.includes("1_lending_club_credit_risk_masterclass|1_lending_club")) {
         l = "[[1_lending_club_credit_risk_masterclass]]";
      }
      return l;
    });

    const uniqueLinks = [...new Set(updatedLinks.filter(Boolean))];

    if (JSON.stringify(data.links) !== JSON.stringify(uniqueLinks)) {
      data.links = uniqueLinks;
      modified = true;
    }
  }

  // 3. Fix all broken links in body content
  const newContent = content
    .replace(/\[\[sandbox_1\]\]/g, '[[2_sandbox]]')
    .replace(/\[\[sandbox_2\]\]/g, '[[3_sandbox]]')
    .replace(/\[\[sandbox_3\]\]/g, '[[4_sandbox]]')
    .replace(/\[\[sandbox_1\|(.*?)\]\]/g, '[[2_sandbox|$1]]')
    .replace(/\[\[sandbox_2\|(.*?)\]\]/g, '[[3_sandbox|$1]]')
    .replace(/\[\[sandbox_3\|(.*?)\]\]/g, '[[4_sandbox|$1]]')
    // old node replacement
    .replace(/\[\[1_full_pd_model\]\]/g, '[[1_lending_club_credit_risk_masterclass]]')
    .replace(/\[\[2_monitoring_model\]\]/g, '[[1_lending_club_credit_risk_masterclass]]')
    .replace(/\[\[3_lgd_ead_model_rewritten\]\]/g, '[[1_lending_club_credit_risk_masterclass]]')
    .replace(/\[\[4_ecl_cecl_stress_testing_rewritten\]\]/g, '[[1_lending_club_credit_risk_masterclass]]')
    // Keep piped text if they used one
    .replace(/\[\[1_full_pd_model\|(.*?)\]\]/g, '[[1_lending_club_credit_risk_masterclass|$1]]')
    .replace(/\[\[2_monitoring_model\|(.*?)\]\]/g, '[[1_lending_club_credit_risk_masterclass|$1]]')
    .replace(/\[\[3_lgd_ead_model_rewritten\|(.*?)\]\]/g, '[[1_lending_club_credit_risk_masterclass|$1]]')
    .replace(/\[\[4_ecl_cecl_stress_testing_rewritten\|(.*?)\]\]/g, '[[1_lending_club_credit_risk_masterclass|$1]]');

  if (newContent !== content) {
    content = newContent;
    modified = true;
  }

  if (modified) {
    const newRaw = matter.stringify(content, data);
    fs.writeFileSync(filePath, newRaw, 'utf-8');
    console.log(`Updated: ${file}`);
  }
}
console.log("Done");

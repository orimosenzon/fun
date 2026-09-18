// Local harness for apps_script/Code.gs: mocks the Apps Script services that
// readCourseSheet() touches and runs it over every tab in a sheets.json export.
//
//     node tools/harness.js [tools/out/sheets.json]
//
// It only exercises the sheet-reading side (parsing, row classification, label
// read/write). Tasks and Calendar calls are not mocked; test those in Apps Script.
const fs = require('fs');
const path = require('path');

const jsonPath = process.argv[2] || path.join(__dirname, 'out', 'sheets.json');
const codePath = path.join(__dirname, '..', 'apps_script', 'Code.gs');
const sheets = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));

function revive(v) {
  if (v && typeof v === 'object' && v.$date) {
    const [y, m, d] = v.$date.split('-').map(Number);
    return new Date(y, m - 1, d);
  }
  return v;
}

function makeSheet(name, data) {
  const values = data.values.map(r => r.map(revive));
  const bgs = data.backgrounds;
  let lastRow = 0, lastCol = 0;
  values.forEach((r, i) => r.forEach((c, j) => {
    if (String(c).trim() !== '') { lastRow = i + 1; lastCol = Math.max(lastCol, j + 1); }
  }));
  const written = [];
  return {
    getName: () => name,
    getSheetId: () => 1234,
    getLastRow: () => lastRow,
    getLastColumn: () => lastCol,
    written,
    getRange(row, col, nRows = 1, nCols = 1) {
      const slice = (grid, fill) => {
        const out = [];
        for (let r = row - 1; r < row - 1 + nRows; r++) {
          const line = [];
          for (let c = col - 1; c < col - 1 + nCols; c++) line.push((grid[r] || [])[c] ?? fill);
          out.push(line);
        }
        return out;
      };
      return {
        getValues: () => slice(values, ''),
        getBackgrounds: () => slice(bgs, '#ffffff'),
        getValue: () => slice(values, '')[0][0],
        setValue: (v) => {
          written.push({ row, col, v });
          values[row - 1] = values[row - 1] || [];
          values[row - 1][col - 1] = v;
        }
      };
    }
  };
}

const TZ = 'Europe/Amsterdam';
global.SpreadsheetApp = { getActive: () => ({ getSpreadsheetTimeZone: () => TZ }) };
global.Session = { getScriptTimeZone: () => TZ };
global.Utilities = {
  formatDate(date, tz, pattern) {
    const y = date.getFullYear(), m = date.getMonth() + 1, d = date.getDate();
    if (pattern === 'yyyy-M-d') return `${y}-${m}-${d}`;
    return `${y}-${String(m).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
  }
};

eval(fs.readFileSync(codePath, 'utf8'));

let failures = 0;
for (const [name, data] of Object.entries(sheets)) {
  const sheet = makeSheet(name, data);
  console.log('\n=== ' + name);
  try {
    const course = readCourseSheet(sheet);
    console.log('start', fmtDay(course.start), 'end', course.end ? fmtDay(course.end) : '-',
                'email', JSON.stringify(course.trainerEmail));
    const trainer = course.rows.filter(rowFilter('trainer'));
    const pa = course.rows.filter(rowFilter('pa'));
    console.log(`rows ${course.rows.length}: trainer ${trainer.length}, pa ${pa.length}, skipped ${course.warnings.length}`);
    trainer.forEach(r => console.log('  [T] ', fmtDay(r.date), r.task));
    pa.forEach(r => console.log('  [PA]', fmtDay(r.date), r.task));
    course.warnings.forEach(w => console.log('  skip:', w));
  } catch (e) {
    console.log('not a course sheet:', e.message);
    failures++;
  }
}
console.log(`\n${Object.keys(sheets).length} tabs, ${failures} not parsed as course sheets`);

/**
 * Course tools: turn a course sheet into a Google Tasks list and a shared
 * trainer calendar with reminders.
 *
 * Expected layout of a course sheet (the PA's existing format):
 *   A1 "Course start" | B1 <date>
 *   A2 "Course end"   | B2 <date>          (optional)
 *   Row 3: headers    Category | Task | Responsibility | Where | Timing | Exact Date | Notes / Links
 *   Row 4+: one task per row. "Exact Date" is a formula relative to B1/B2.
 *   Rows highlighted yellow (or with Responsibility containing "trainer") are the trainer's.
 *
 * Bound to the spreadsheet: Extensions -> Apps Script. Adds a "Course tools" menu.
 */

var CONFIG = {
  HEADER_ROW: 3,
  COLS: {
    category: 'Category',
    task: 'Task',            // falls back to column B when the header is missing
    responsibility: 'Responsibility',
    where: 'Where',
    timing: 'Timing',
    date: 'Exact Date',
    notes: 'Notes / Links'
  },
  START_LABEL: 'Course start',
  END_LABEL: 'Course end',
  TRAINER_EMAIL_LABEL: 'Trainer email',   // looked up in rows 1-2; written to H1/I1 if missing
  HIGHLIGHT_COLORS: ['#ffff00'],           // yellow = trainer row

  // Which rows go where: 'all' | 'pa' | 'trainer'
  TASKS_ROWS: 'all',
  CALENDAR_ROWS: 'trainer',

  // Reminders before each deadline, in days, fired at REMINDER_HOUR (days >= 1)
  EMAIL_REMINDER_DAYS: [3, 1],
  POPUP_REMINDER_DAYS: [1],
  REMINDER_HOUR: 9,

  // false = share the calendar only (the PA's current practice).
  // true  = also invite the trainer as a guest on every event, so the events land on the
  //         trainer's own calendar and the trainer's own reminders apply.
  INVITE_TRAINER_AS_GUEST: false,
  SHARE_CALENDAR_ROLE: 'writer',           // 'reader' | 'writer'
  ADD_COURSE_BOUNDS: true,                 // add "Course starts" / "Course ends" events

  TAG_KEY: 'courseSheetId'                 // tag on events created by this script
};

// ---------------------------------------------------------------- menu

function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('Course tools')
    .addItem('Create Google Tasks list', 'createTasksList')
    .addItem('Create trainer calendar', 'createTrainerCalendar')
    .addItem('Create both', 'createBoth')
    .addSeparator()
    .addItem('Preview (no changes)', 'previewCourse')
    .addToUi();
}

function createBoth() {
  createTasksList();
  createTrainerCalendar();
}

function previewCourse() {
  var course = readCourseSheet(SpreadsheetApp.getActiveSheet());
  var lines = ['Course: ' + course.name,
               'Start: ' + fmtDay(course.start) + '   End: ' + (course.end ? fmtDay(course.end) : '-'),
               'Trainer email: ' + (course.trainerEmail || '(not set)'),
               ''];
  course.rows.forEach(function (r) {
    lines.push((r.isTrainer ? '[Trainer] ' : '[PA] ') + fmtDay(r.date) + '  ' + r.task);
  });
  if (course.warnings.length) lines.push('', 'Skipped:', course.warnings.join('\n'));
  SpreadsheetApp.getUi().alert(lines.join('\n'));
}

// ---------------------------------------------------------------- Google Tasks

function createTasksList() {
  var ui = SpreadsheetApp.getUi();
  var course = readCourseSheet(SpreadsheetApp.getActiveSheet());
  var rows = course.rows.filter(rowFilter(CONFIG.TASKS_ROWS));
  if (!rows.length) return ui.alert('No dated tasks found on this sheet.');

  var existing = findTasksList(course.name);
  if (existing) {
    var ans = ui.alert('Tasks list "' + course.name + '" already exists',
                       'Delete it and create it again from the sheet?', ui.ButtonSet.YES_NO);
    if (ans !== ui.Button.YES) return;
    Tasks.Tasklists.remove(existing.id);
  }

  var list = Tasks.Tasklists.insert({ title: course.name });
  var previousId = null;
  rows.forEach(function (r) {
    var task = {
      title: (r.isTrainer && CONFIG.TASKS_ROWS === 'all' ? '[Trainer] ' : '') + r.task,
      notes: taskNotes(r),
      due: fmtDay(r.date) + 'T00:00:00.000Z'   // Tasks keeps the date only
    };
    var created = Tasks.Tasks.insert(task, list.id, previousId ? { previous: previousId } : {});
    previousId = created.id;                     // keeps the sheet's order
  });

  ui.alert('Created Tasks list "' + course.name + '" with ' + rows.length + ' tasks.' +
           warningsText(course));
}

function findTasksList(title) {
  var res = Tasks.Tasklists.list({ maxResults: 100 });
  var items = (res && res.items) || [];
  for (var i = 0; i < items.length; i++) if (items[i].title === title) return items[i];
  return null;
}

// ---------------------------------------------------------------- Calendar

function createTrainerCalendar() {
  var ui = SpreadsheetApp.getUi();
  var sheet = SpreadsheetApp.getActiveSheet();
  var course = readCourseSheet(sheet);
  var rows = course.rows.filter(rowFilter(CONFIG.CALENDAR_ROWS));
  if (!rows.length) return ui.alert('No dated trainer tasks found on this sheet.');

  if (!course.trainerEmail) {
    var resp = ui.prompt('Trainer email', 'Email address to share the calendar with (leave empty to skip sharing):',
                         ui.ButtonSet.OK_CANCEL);
    if (resp.getSelectedButton() !== ui.Button.OK) return;
    course.trainerEmail = resp.getResponseText().trim();
    writeLabelValue(sheet, CONFIG.TRAINER_EMAIL_LABEL, course.trainerEmail);
  }

  var cal = findOrCreateCalendar(sheet, course.name);
  var removed = removeScriptEvents(cal, sheet);   // re-run = rebuild the events

  var tz = SpreadsheetApp.getActive().getSpreadsheetTimeZone();
  var created = 0;
  rows.forEach(function (r) {
    addDeadlineEvent(cal, sheet, r.task, r.date, eventDescription(r), course.trainerEmail);
    created++;
  });
  if (CONFIG.ADD_COURSE_BOUNDS) {
    addDeadlineEvent(cal, sheet, 'Course starts: ' + course.name, course.start, '', course.trainerEmail, true);
    if (course.end) addDeadlineEvent(cal, sheet, 'Course ends: ' + course.name, course.end, '', course.trainerEmail, true);
  }

  var shared = '';
  if (course.trainerEmail) shared = shareCalendar(cal.getId(), course.trainerEmail);

  var link = 'https://calendar.google.com/calendar/u/0/r?cid=' + encodeURIComponent(cal.getId());
  writeLabelValue(sheet, 'Calendar', link);

  ui.alert('Calendar "' + course.name + '" (' + tz + '): ' + created + ' deadline events' +
           (removed ? ' (replaced ' + removed + ' old events)' : '') + '.\n' + shared +
           warningsText(course));
}

function findOrCreateCalendar(sheet, name) {
  var props = PropertiesService.getDocumentProperties();
  var key = 'calendar:' + sheet.getSheetId();
  var id = props.getProperty(key);
  var cal = id ? CalendarApp.getCalendarById(id) : null;
  if (cal) return cal;
  cal = CalendarApp.createCalendar(name, {
    summary: 'Deadlines for ' + name + ' (generated from the course sheet)',
    timeZone: SpreadsheetApp.getActive().getSpreadsheetTimeZone()
  });
  props.setProperty(key, cal.getId());
  return cal;
}

function removeScriptEvents(cal, sheet) {
  var tag = String(sheet.getSheetId());
  var events = cal.getEvents(new Date(2000, 0, 1), new Date(2100, 0, 1));
  var n = 0;
  events.forEach(function (e) {
    if (e.getTag(CONFIG.TAG_KEY) === tag) { e.deleteEvent(); n++; }
  });
  return n;
}

function addDeadlineEvent(cal, sheet, title, date, description, guestEmail, noReminders) {
  var opts = { description: description || '' };
  if (CONFIG.INVITE_TRAINER_AS_GUEST && guestEmail) {
    opts.guests = guestEmail;
    opts.sendInvites = true;
  }
  var ev = cal.createAllDayEvent(title, date, opts);
  ev.setTag(CONFIG.TAG_KEY, String(sheet.getSheetId()));
  ev.removeAllReminders();
  if (!noReminders) {
    CONFIG.EMAIL_REMINDER_DAYS.forEach(function (d) { ev.addEmailReminder(minutesBefore(d)); });
    CONFIG.POPUP_REMINDER_DAYS.forEach(function (d) { ev.addPopupReminder(minutesBefore(d)); });
  }
  return ev;
}

// All-day events start at midnight; "d days before at REMINDER_HOUR" in minutes.
// CalendarApp accepts 5 minutes to 4 weeks, so days must be >= 1.
function minutesBefore(days) {
  var m = days * 24 * 60 - CONFIG.REMINDER_HOUR * 60;
  if (m < 5 || m > 40320) throw new Error('Reminder of ' + days + ' day(s) before is out of range (1-28).');
  return m;
}

function shareCalendar(calendarId, email) {
  var acl = Calendar.Acl.list(calendarId);
  var items = (acl && acl.items) || [];
  for (var i = 0; i < items.length; i++) {
    if (items[i].scope && items[i].scope.value === email) return 'Already shared with ' + email + '.';
  }
  Calendar.Acl.insert({ role: CONFIG.SHARE_CALENDAR_ROLE, scope: { type: 'user', value: email } }, calendarId);
  return 'Shared with ' + email + ' (' + CONFIG.SHARE_CALENDAR_ROLE + ').';
}

// ---------------------------------------------------------------- reading the sheet

function readCourseSheet(sheet) {
  var name = sheet.getName();
  var start = findLabelValue(sheet, CONFIG.START_LABEL);
  if (!(start instanceof Date)) {
    throw new Error('Cell next to "' + CONFIG.START_LABEL + '" must be a date. Is this a course sheet?');
  }
  var end = findLabelValue(sheet, CONFIG.END_LABEL);
  if (!(end instanceof Date)) end = null;
  var trainerEmail = findLabelValue(sheet, CONFIG.TRAINER_EMAIL_LABEL);
  trainerEmail = trainerEmail ? String(trainerEmail).trim() : '';

  var lastRow = sheet.getLastRow();
  var lastCol = sheet.getLastColumn();
  var headers = sheet.getRange(CONFIG.HEADER_ROW, 1, 1, lastCol).getValues()[0].map(function (h) {
    return String(h).trim().toLowerCase();
  });
  var col = {};
  Object.keys(CONFIG.COLS).forEach(function (k) {
    col[k] = headers.indexOf(CONFIG.COLS[k].toLowerCase());
  });
  if (col.task < 0) col.task = 1;                         // "Assistants Procedure" has no Task header
  if (col.date < 0) throw new Error('No "' + CONFIG.COLS.date + '" column in row ' + CONFIG.HEADER_ROW + '.');

  var first = CONFIG.HEADER_ROW + 1;
  var n = lastRow - first + 1;
  if (n <= 0) return { name: name, start: start, end: end, trainerEmail: trainerEmail, rows: [], warnings: [] };
  var range = sheet.getRange(first, 1, n, lastCol);
  var values = range.getValues();
  var colors = range.getBackgrounds();

  var rows = [], warnings = [], started = false;
  for (var i = 0; i < n; i++) {
    var v = values[i];
    var empty = v.every(function (c) { return String(c).trim() === ''; });
    if (empty) {
      if (started) break;                                 // table ends at the first blank row
      continue;                                           // but may start a few rows below the header
    }
    started = true;
    var cell = function (k) { return col[k] >= 0 ? String(v[col[k]]).trim() : ''; };
    var task = cell('task');
    if (!task) continue;
    var date = col.date >= 0 ? v[col.date] : null;
    if (!(date instanceof Date)) {
      warnings.push('row ' + (first + i) + ': "' + task + '" has no date (' + cell('timing') + ')');
      continue;
    }
    var isYellow = CONFIG.HIGHLIGHT_COLORS.indexOf(String(colors[i][col.task]).toLowerCase()) >= 0;
    var resp = cell('responsibility');
    rows.push({
      rowNum: first + i,
      task: task,
      category: cell('category'),
      responsibility: resp,
      where: cell('where'),
      timing: cell('timing'),
      notes: cell('notes'),
      date: toLocalDay(date),
      isTrainer: isYellow || /trainer/i.test(resp)
    });
  }
  return { name: name, start: toLocalDay(start), end: end ? toLocalDay(end) : null,
           trainerEmail: trainerEmail, rows: rows, warnings: warnings };
}

// Value of the cell to the right of a label found in rows 1-2 (A1:Z2), or ''.
function findLabelValue(sheet, label) {
  var v = sheet.getRange(1, 1, 2, 26).getValues();
  for (var r = 0; r < v.length; r++) {
    for (var c = 0; c < v[r].length - 1; c++) {
      if (String(v[r][c]).trim().toLowerCase() === label.toLowerCase()) return v[r][c + 1];
    }
  }
  return '';
}

// Writes label/value next to an existing label, or into the first free H/I slot in rows 1-2.
function writeLabelValue(sheet, label, value) {
  var v = sheet.getRange(1, 1, 2, 26).getValues();
  for (var r = 0; r < v.length; r++) {
    for (var c = 0; c < v[r].length - 1; c++) {
      if (String(v[r][c]).trim().toLowerCase() === label.toLowerCase()) {
        sheet.getRange(r + 1, c + 2).setValue(value);
        return;
      }
    }
  }
  for (var row = 1; row <= 2; row++) {
    if (String(sheet.getRange(row, 8).getValue()).trim() === '') {
      sheet.getRange(row, 8).setValue(label);
      sheet.getRange(row, 9).setValue(value);
      return;
    }
  }
}

function rowFilter(mode) {
  if (mode === 'trainer') return function (r) { return r.isTrainer; };
  if (mode === 'pa') return function (r) { return !r.isTrainer; };
  return function () { return true; };
}

// Sheet dates are midnight in the spreadsheet time zone; rebuild the same calendar day at noon
// in the script time zone so DST and time-zone differences cannot shift the day.
function toLocalDay(date) {
  var tz = SpreadsheetApp.getActive().getSpreadsheetTimeZone();
  var parts = Utilities.formatDate(date, tz, 'yyyy-M-d').split('-').map(Number);
  return new Date(parts[0], parts[1] - 1, parts[2], 12, 0, 0);
}

function fmtDay(date) {
  return Utilities.formatDate(date, Session.getScriptTimeZone(), 'yyyy-MM-dd');
}

function taskNotes(r) {
  return [r.responsibility && 'Responsibility: ' + r.responsibility,
          r.where && 'Where: ' + r.where,
          r.timing && 'Timing: ' + r.timing,
          r.notes && 'Notes: ' + r.notes].filter(Boolean).join('\n');
}

function eventDescription(r) {
  return [r.category && 'Category: ' + r.category,
          r.where && 'Where: ' + r.where,
          r.timing && 'Timing: ' + r.timing,
          r.notes && r.notes].filter(Boolean).join('\n');
}

function warningsText(course) {
  return course.warnings.length ? '\n\nSkipped (no date):\n' + course.warnings.join('\n') : '';
}

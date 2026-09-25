"""watch_objections.py: a week of mornings, against made-up answers.

No network. Each "day" writes the service's answer to a file and runs the
script with --today and --fixture, so what is tested is the real command line
the cron job runs, state file and all.
"""
import datetime as dt, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), 'watch_objections.py')

fails = []
def check(name, ok, detail=''):
    print(('  ok   ' if ok else '  FAIL ') + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)

def ms(iso):
    return int(dt.datetime.fromisoformat(iso).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)

def plan(num, shuts=None, units=0, status='פרסום הפקדה'):
    return {'pl_number': num, 'pl_name': 'תכנית ' + num, 'pl_url': 'https://mavat.example/' + num,
            'internet_short_status': status, 'quantity_delta_120': units,
            'pl_last_deposit_date': ms(shuts) if shuts else None,
            'pl_rejection_date': None, 'pl_date_advertise': None}

tmp = tempfile.mkdtemp()
state = os.path.join(tmp, 'state.json')
reports = os.path.join(tmp, 'reports')

def day(today, plans):
    fx = os.path.join(tmp, 'fx.json')
    json.dump({'features': [{'attributes': p} for p in plans]}, open(fx, 'w'))
    out = subprocess.run([sys.executable, SCRIPT, '--fixture', fx, '--state', state,
                          '--reports', reports, '--today', today],
                         capture_output=True, text=True)
    if out.returncode:
        print(out.stderr)
    news = int(next(l for l in out.stdout.splitlines() if l.startswith('NEWS='))[5:])
    path = os.path.join(reports, today + '.md')
    text = open(path, encoding='utf-8').read() if os.path.exists(path) else None
    return news, text

# Day 1: first run. A open until 10/1, B deposited with no date yet.
news, text = day('2026-01-01', [plan('A', '2026-01-10', 12), plan('B')])
check('first run writes a report', text is not None)
check('first run calls nothing new', news == 0, news)
check('first run lists what is open', 'תכנית A' in text and '12 יחידות דיור' in text, text)

# Day 2: nothing changed, A is 8 days out.
news, text = day('2026-01-02', [plan('A', '2026-01-10', 12), plan('B')])
check('a quiet day writes nothing', news == 0 and text is None, (news, text))

# Day 3: A is 7 days out, B gets a date, C enters deposit.
news, text = day('2026-01-03', [plan('A', '2026-01-10', 12), plan('B', '2026-03-01'), plan('C')])
check('three pieces of news', news == 3, news)
check('B opened', text and '## נפתח חלון התנגדות\n\n- **תכנית B**' in text, text)
check('A is closing in 7', text and 'עומד להיסגר' in text and 'בעוד 7 ימים' in text, text)
check('C deposited without a date', text and 'הופקדה, עוד בלי תאריך' in text and 'תכנית C' in text, text)

# Day 4: A at 6 days: the 7-day reminder was already said.
news, text = day('2026-01-04', [plan('A', '2026-01-10', 12), plan('B', '2026-03-01'), plan('C')])
check('no second reminder for the same threshold', news == 0, news)

# Day 9: A tomorrow. Skipped days 5-8 (no run), 3 and 1 both due: one mention.
news, text = day('2026-01-09', [plan('A', '2026-01-10', 12), plan('B', '2026-03-01'), plan('C')])
check('missed thresholds fold into one mention', news == 1 and 'מחר' in text, (news, text))

# Day 10: last day.
news, text = day('2026-01-10', [plan('A', '2026-01-10', 12), plan('B', '2026-03-01'), plan('C')])
check('last day is said', news == 1 and 'היום' in text, (news, text))

# Day 11: A has shut (the service still lists it in deposit).
news, text = day('2026-01-11', [plan('A', '2026-01-10', 12), plan('B', '2026-03-01'), plan('C')])
check('A closed', news == 1 and '## נסגר' in text and 'נסגר ב-10/01/2026' in text, (news, text))

# Day 12: B is extended while open.
news, text = day('2026-01-12', [plan('A', '2026-01-10', 12), plan('B', '2026-03-15'), plan('C')])
check('an extension is news', news == 1 and 'תכנית B' in text and '15/03/2026' in text, (news, text))

# Day 13: B drops out of deposit while its window is open.
news, text = day('2026-01-13', [plan('A', '2026-01-10', 12), plan('C')])
check('a plan that vanishes while open is said', news == 1 and 'יצאה משלב ההפקדה' in text, (news, text))

print()
print('FAIL: ' + ', '.join(fails) if fails else 'all ok')
sys.exit(1 if fails else 0)

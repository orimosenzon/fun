import json,glob,sys
files = sys.argv[1:] or sorted(glob.glob('mavat/*.json'))
for fn in files:
    d=json.load(open(fn)); p=d.get('planDetails') or {}
    print('\n\n##########', p.get('NUMB'), p.get('E_NAME'), '| auth', p.get('AUTH'), '|', fn)
    print('GOALS:', p.get('GOALS')); print('INSTR:', p.get('INSTRACTIONS'))
    if d.get('recExplanation'): print('EXPL:', json.dumps(d['recExplanation'],ensure_ascii=False)[:3000])
    print('BLOCKS:', [(b['BLOCKS'],b['PARCELS_WHOLE'],b['PARCELS_PARTIAL']) for b in d.get('rsBlocks') or []])
    print('LOC:', d.get('locationDesc'), [ (r.get('STREET_NAME'),r.get('HOUSE_NUMBER')) for r in d.get('rsLocation') or []])
    for r in d.get('rsRelation') or []: print('REL:', r['RELATION_TYPE'], r['PLAN_NUMBER'], r['PLAN_NAME'], r.get('PUBLICATION_DATE'), 'mp', r.get('ORG_MP_ID'))
    for r in d.get('rsTopic') or []: print('CHANGED-BY:', r.get('ORG_N'), r.get('RELATION_TYPE'), 'mp', r.get('ORG_MP_ID'))
    for q in d.get('rsQuantities') or []: print('Q:', q['QUANTITY_DESC'], q['AUTHORISED_QUANTITY'], q.get('AUTHORISED_QUANTITY_ADD'), q.get('REMARK'))
    for r in d.get('rsDes') or []: print('DECISION', r['MEETING_DATE'], r['FO_NAME'], ':', (r.get('MMI_DESICIONS') or '')[:2500])
    for r in d.get('rsOppositions') or []: print('OPP:', json.dumps(r,ensure_ascii=False)[:300])
    for r in d.get('rsInternet') or []: print('  STEP', r['EIS_DATE'], r['LIS_DESC'], (r.get('DETAILS') or '')[:150])
    for r in d.get('rsLocalPlanActions') or []: print('LOCAL:', json.dumps(r,ensure_ascii=False)[:300])

import sys,zipfile,re,html,glob,os
os.makedirs('../sections',exist_ok=True)
def docx(fn):
    x=zipfile.ZipFile(fn).read('word/document.xml').decode()
    x=re.sub(r'</w:p>','\n',x); t=html.unescape(re.sub(r'<[^>]+>','',x)); return re.sub(r'\n\s*\n+','\n',t)
for fn in sorted(glob.glob('*_10.docx')):
    plan=fn.split('__')[0].replace('ש_','ש/ ')
    t=docx(fn)
    pat=re.escape(plan).replace('\\ ','\\s*')
    starts=[m.start() for m in re.finditer(r'\n\d+\.\s*תו?כנית\s*-\s*'+pat, t)]
    if not starts: starts=[m.start() for m in re.finditer(pat,t)][1:2]
    if not starts: print('nosec',fn); continue
    s=starts[0]; m=re.search(r'\n\d+\.\s*תו?כנית\s*-', t[s+5:])
    sec=t[s:s+5+m.start()] if m else t[s:]
    open('../sections/'+fn.replace('.docx','.txt'),'w').write(sec); print(len(sec),fn)

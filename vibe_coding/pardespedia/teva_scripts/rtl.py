import subprocess,re,sys,html
def page_lines(pdf,p):
    x=subprocess.run(['pdftotext','-bbox','-f',str(p),'-l',str(p),pdf,'-'],capture_output=True,text=True).stdout
    ws=[(float(a),float(b),float(c),float(d),html.unescape(t)) for a,b,c,d,t in re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>',x)]
    ws=[(a,b,c,d,re.sub(r'[‪-‮]','',t)) for a,b,c,d,t in ws if t.strip()]
    ws.sort(key=lambda w:(round((w[1]+w[3])/2/3),-w[0]))
    lines=[];cur=[];cy=None
    for w in sorted(ws,key=lambda w:(w[1]+w[3])/2):
        y=(w[1]+w[3])/2
        if cy is None or abs(y-cy)>3: 
            if cur: lines.append(cur)
            cur=[];cy=y
        cur.append(w)
    if cur: lines.append(cur)
    out=[]
    for ln in lines:
        ln.sort(key=lambda w:-w[2])
        s='';prev=None
        for w in ln:
            if prev is not None:
                gap=prev[0]-w[2]
                h=w[3]-w[1]
                if gap>h*0.22: s+=' ' if gap<h*4 else '   |   '
            s+=w[4][::-1] if len(w[4])>1 and not re.search(r'[A-Za-z0-9]',w[4]) else w[4]
            prev=w
        # digit/latin runs come out reversed
        s=re.sub(r'[0-9A-Za-z][0-9A-Za-z.,/%\-]*[0-9A-Za-z]',lambda m:m.group(0)[::-1],s)
        out.append((ln[0][1],s))
    return out
if __name__=='__main__':
    for y,s in page_lines(sys.argv[1],int(sys.argv[2])): print(s)

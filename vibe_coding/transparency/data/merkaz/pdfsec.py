import re,sys,glob
BIDI=re.compile('[‎‏‪-‮]')
def clean(t): return BIDI.sub('',t)
def section(fn, key, n=12000):
    t=clean(open(fn).read())
    occ=[m.start() for m in re.finditer(key,t)]
    if not occ: return ''
    # the body starts at the occurrence followed by "מטרת הדיון" or "מוזמנים"/"החלטה" soonest; use the last-but agenda occurrence
    s=occ[1] if len(occ)>1 else occ[0]
    return t[s:s+n]
if __name__=='__main__':
    print(section(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv)>3 else 12000))

import json,os
D=os.path.expanduser('~/.cache/pardespedia_teva/articles')
def w(key,**k):
    json.dump(k,open(f'{D}/{key}.json','w'),ensure_ascii=False,indent=1)

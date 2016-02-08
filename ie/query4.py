import requests

apiUrl = 'http://en.wikipedia.org/w/api.php'

# == all of a category == 
def membersInCategory(cname):
    params = {'action': 'query', 'format': 'json', 'list':'categorymembers'}
    params['cmtitle']='Category:'+cname
#    apiRequest(params)
    query(params)

# == all templates in page == 
def templatesInPage(pname):
    params = {'action':'query', 'format':'json', 'prop':'templates'}
    params['titles'] =  pname 
    query(params)

# == Query == 
def apiRequest(params): 
    result = requests.get(apiUrl, params = params).json()
    print(result)

def query(params):
    for item in contQuery(params):
        print(item) 

def contQuery(params):
    params['generator'] = 'allpages'
    lastContinue = {'continue': ''}
    while True:
        req = params.copy()

        req.update(lastContinue)

        result = requests.get(apiUrl, params = params).json()

        if 'error' in result: raise Error(result['error'])
        if 'warnings' in result: print(result['warnings'])
        if 'query' in result: yield result['query']
        if 'continue' not in result: break
        lastContinue = result['continue']    

# == main ==

#membersInCategory('21st-century_American_male_actors')
#membersInCategory('American_male_voice_actors')


templatesInPage("Brad_pitt")













# == a page's content == 
## https://en.wikipedia.org/w/api.php?action=query&titles=almog&rvprop=content&prop=revisions

## == all categories of a page == 
## https://en.wikipedia.org/w/api.php?action=query&titles=Almog&prop=categories

## == list of members in catedgory == 
## https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:Kibbutzim

## == all 21 centurary male actors ==  
## https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:21st-century_American_male_actors

## == all templates used in a page == 
## https://en.wikipedia.org/w/api.php?action=query&titles=almog&prop=templates

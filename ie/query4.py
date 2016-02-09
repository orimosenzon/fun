import requests

apiUrl = 'http://en.wikipedia.org/w/api.php'

# == all of a category == 
def membersInCategory(cname):
    params = {'action': 'query', 'format': 'json', 'list':'categorymembers'}
    params['cmtitle']='Category:'+cname
    for item in contQuery(params):
        mems = item['categorymembers']
        for mem in mems:
            print(mem['title'])
        print('\n\n  ** end of chunck ** \n\n')    

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
        print('\n\n  ** end of chunck ** \n\n') 

def contQuery(params):
    params['generator'] = 'allpages'
    lastContinue = {'continue': ''}
    while True:
        params1 = params.copy()

        params1.update(lastContinue)

        result = requests.get(apiUrl, params = params1).json()

        if 'error' in result: raise Error(result['error'])
        if 'warnings' in result: print("\n\n *Warning* \n\n",result['warnings'])
        if 'query' in result: yield result['query']
        if 'continue' not in result: break
        lastContinue = result['continue']    

# == main ==

membersInCategory('21st-century_American_male_actors')
#membersInCategory('American_male_voice_actors')


#templatesInPage("Brad_pitt")










# == header attempt ==
##headers = {'user_agent':'MyCoolTool/1.1 (https://example.org/MyCoolTool/; MyCoolTool@example.org) BasedOnSuperLib/1.4'}
##requests.post(apiUrl, headers=headers)

##headers = {'Api-User-Agent':'RExample/1.0'}
##requests.post(apiUrl, headers=headers)


##headers = {'content-type': 'application/x-www-form-urlencoded'}
##r4 = requests.post(apiUrl, headers=headers)
##print (r4.text)



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

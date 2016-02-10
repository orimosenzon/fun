import requests
import mwparserfromhell

apiUrl = 'http://en.wikipedia.org/w/api.php'

def dates(name):
    params = {'action': 'query', 'prop':'revisions', 'rvprop':'content','rvsection':'0','format': 'json'}
    params['titles'] = name
    result = requests.get(apiUrl, params = params).json()
    text = list(result["query"]["pages"].values())[0]["revisions"][0]["*"]

    wiki = mwparserfromhell.parse(text)

    birth_data = wiki.filter_templates(matches="Birth date")[1]
    birth_year = birth_data.get(1).value
    birth_month = birth_data.get(2).value
    birth_day = birth_data.get(3).value

    death_data = wiki.filter_templates(matches="Death date")[1]
    death_year = death_data.get(1).value
    death_month = death_data.get(2).value
    death_day = death_data.get(3).value

    print("born: ",birth_day,".",birth_month,".",birth_year)
    print("died: ",death_day,".",death_month,".",death_year)

# == all of a category == 
def membersInCategory(cname):
    params = {'action': 'query', 'format': 'json', 'list':'categorymembers'}
    params['cmtitle']='Category:'+cname

    for item in contQuery(params):
        mems = item['categorymembers']
        for mem in mems:
            name = mem['title']
            print(name)
##            try:
##                dates(name)
##            except Exception:
##                pass
        print('\n\n  ** end of chunck ** \n\n')    

# == all templates in page == 
def templatesInPage(pname):
    params = {'action':'query', 'format':'json', 'prop':'templates'}
    params['titles'] =  pname

    query(params)

# == Query == 

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

#membersInCategory('Kibbutzim')
membersInCategory('21st-century_American_male_actors')
#membersInCategory('American_male_voice_actors')

#templatesInPage("Brad_pitt")

#dates('Albert Einstein')






## == leftovers == 

# == ==
# result['query']['pages']['1317804']['templates']

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


# help sources:
# http://stackoverflow.com/questions/12250580/parse-birth-and-death-dates-from-wikipedia

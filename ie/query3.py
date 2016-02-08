import requests

url = 'http://en.wikipedia.org/w/api.php'

params = {'action': 'query', 'format': 'json', 'prop':'revisions'}
params['rvprop'] = 'content'
params['titles'] = 'Almog'
#params['rvsection']=3

result = requests.get(url, params = params).json() 

print(result)
    



# a page's content
## https://en.wikipedia.org/w/api.php?action=query&titles=almog&rvprop=content&prop=revisions

## all categories of a page 
## https://en.wikipedia.org/w/api.php?action=query&titles=Almog&prop=categories

## list of members in catedgory:
## https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:Kibbutzim

## all 21 centurary male actors: 
## https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:21st-century_American_male_actors

## all templates used in a page:
## https://en.wikipedia.org/w/api.php?action=query&titles=almog&prop=templates

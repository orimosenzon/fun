import requests
import mwparserfromhell

url = 'http://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&rvsection=0&titles=Albert_Einstein&format=json'

res = requests.get(url).json()
text = list(res["query"]["pages"].values())[0]["revisions"][0]["*"]
wiki = mwparserfromhell.parse(text)

##for template in wiki.filter_templates(matches="Birth date"):
##    print(template)
##    print("\n**********\n\n")

    
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

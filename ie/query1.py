import urllib.request

urlBase = "https://en.wikipedia.org/w/api.php?"
#urlParam = "action=query&prop=revisions&rvprop=content&rvsection=0&titles=Brad_Pitt&format=json"
urlParam = "action=query&list=allcategories&acprefix=List%20of&continue=&format=json"
url  = urlBase+urlParam 

f = urllib.request.urlopen(url)

s = f.read()

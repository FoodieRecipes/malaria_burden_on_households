import urllib.request
url = 'https://ee.kobotoolbox.org/x/js/build/chunks/chunk-CWON3Y5J.js'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    js = resp.read().decode('utf-8', errors='ignore')

idx = js.find('GF=(e,t)=>{')
print(js[idx:idx+300])


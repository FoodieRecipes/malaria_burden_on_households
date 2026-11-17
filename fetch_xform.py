import urllib.request
import urllib.parse
import json

url = 'https://ee.kobotoolbox.org/transform/xform/hsq9SGTk'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'Referer': 'https://ee.kobotoolbox.org/x/hsq9SGTk',
    'X-Requested-With': 'XMLHttpRequest'
}
data = urllib.parse.urlencode({}).encode('utf-8')

req = urllib.request.Request(url, data=data, headers=headers)
try:
    with urllib.request.urlopen(req) as resp:
        res = resp.read().decode('utf-8')
        print(f'Status: {resp.status}')
        print(f'Response length: {len(res)}')
        with open('xform_response.json', 'w', encoding='utf-8') as f:
            f.write(res)
        data_json = json.loads(res)
        print('JSON keys:', list(data_json.keys()))
        if 'form' in data_json:
            print('Form length:', len(data_json['form']))
            with open('xform_form.html', 'w', encoding='utf-8') as f:
                f.write(data_json['form'])
        if 'model' in data_json:
            print('Model length:', len(data_json['model']))
            with open('xform_model.xml', 'w', encoding='utf-8') as f:
                f.write(data_json['model'])
except Exception as e:
    print('POST request failed:', e)


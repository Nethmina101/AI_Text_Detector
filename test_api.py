import urllib.request
import urllib.parse
import sys

try:
    data = urllib.parse.urlencode({'text_input': 'This is a sample test document for testing.'}).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:5000/analyze', data=data)
    response = urllib.request.urlopen(req)
    print("SUCCESS", response.status)
    print(response.read().decode('utf-8')[:200])
except Exception as e:
    import traceback
    traceback.print_exc()

import urllib.request
import re

url = "https://huggingface.co/datasets/allenai/qasper/raw/main/qasper.py"
data = urllib.request.urlopen(url).read().decode('utf-8')
urls = re.findall(r'https?://[^\s\"\'\)]+', data)
for u in set(urls):
    print(u)

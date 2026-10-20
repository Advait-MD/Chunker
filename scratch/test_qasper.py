import urllib.request
import tarfile
import io
import json

url = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    tf = tarfile.open(fileobj=io.BytesIO(resp.read()), mode="r:gz")
    print("Files in tar:", tf.getnames())
    f = tf.extractfile("qasper-dev-v0.3.json")
    data = json.load(f)
    print("Papers in dev:", len(data))
    first_key = list(data.keys())[0]
    paper = data[first_key]
    print("Paper 1 title:", paper.get('title'))
    print("QAs count:", len(paper.get('qas', [])))

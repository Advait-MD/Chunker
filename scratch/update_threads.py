with open('face.py', 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace('cpu_threads = min(8, os.cpu_count() or 4)', 'cpu_threads = 4')
c = c.replace('"num_predict": 50', '"num_predict": 45')

with open('face.py', 'w', encoding='utf-8') as f:
    f.write(c)

print("Updated face.py threads and num_predict successfully!")

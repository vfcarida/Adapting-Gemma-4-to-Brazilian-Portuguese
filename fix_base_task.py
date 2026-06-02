import codecs

with codecs.open('src/eval/tasks/base_task.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

content = content.replace('Ac', 'é')
content = content.replace('PadrAo', 'Padrão')
content = content.replace('parAnteses', 'parênteses')

with codecs.open('src/eval/tasks/base_task.py', 'w', encoding='utf-8') as f:
    f.write(content)

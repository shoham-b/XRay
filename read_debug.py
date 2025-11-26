try:
    with open('debug_output.txt', 'r', encoding='utf-16le') as f:
        print(f.read())
except:
    with open('debug_output.txt', 'r', encoding='utf-8') as f:
        print(f.read())

try:
    with open('debug_contour_90.txt', 'r', encoding='utf-16le') as f:
        print(f.read())
except:
    with open('debug_contour_90.txt', 'r', encoding='utf-8') as f:
        print(f.read())

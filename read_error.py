with open('error_details.txt', 'r') as f:
    content = f.read()
    print(content)
    print("\n" + "="*50)
    print(f"Total characters: {len(content)}")

import ollama

response = ollama.generate(
    model='gemma:2b',
    prompt='What is a qubit?'
)

print(response['response'])

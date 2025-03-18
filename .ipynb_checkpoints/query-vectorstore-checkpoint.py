#! /bin/env python3

# my imports to query stuff...
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# create embedding object... (change host if needed)
ollama_embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url='http://10.100.200.57:11434'
)

persist_directory = "vectorstore" # Path to your saved Chroma store
chroma_store = Chroma(
    embedding_function=ollama_embeddings,
    persist_directory='vectorstore',
)

query = input("Enter your prompt: ")

# query = "Tell me about the mayor"

results = chroma_store.similarity_search(query,k=6)
print("\n\n")
for result in results:
    print("\n--------------------")
    print(result.page_content)
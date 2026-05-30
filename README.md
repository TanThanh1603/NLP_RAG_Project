# download ollama
https://ollama.com/

# set up models
model-generation: qwen2.5-coder:7b
model-embeddings: nomic-embed-text

# run ollama models
ollama serve

# run python
python -m chainlit run RAG_Project.py -w
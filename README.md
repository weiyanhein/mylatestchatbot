There are  few steps to run this chat bot .

1.we need to install local ollama ,mxbai-embed-large:335m   is for embedding and   llama3:8b  is local LLM 
2.we need to install mongodb to store data .
3.we need to run requirements.txt 
4.we need to run the srcipt  prepare_data.py to establish our data sources and to make required vector database that use chromadb 
5.then run uvicorn backend.app:app --reload and then make  testing from swagger UI like {"user_message":"I need a serum product"}


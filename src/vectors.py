import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from scrape import scraped_content
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import numpy as np 


def chunk_data(contents_dict, chunk_size=2000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_contents = {}
    for url, content in contents_dict.items():
        chunked_content = text_splitter.split_text(content)
        chunked_contents[url] = chunked_content
    return chunked_contents

chunked_content = chunk_data(scraped_content)
for url, chunks in chunked_content.items():
    print(f"URL: {url}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(chunk[:200] + '...')  


def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

index_name = "modelindex"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 

index = pc.Index(index_name)


# Upsert vectors into the index
for url, chunks in chunked_content.items():
    for i, chunk in enumerate(chunks):
        chunk_embedding = embeddings.embed_query(chunk)
        print('length',len(chunk_embedding))
        index.upsert(
            vectors=[{"id": f"{url}_chunk_{i}", "values": chunk_embedding}],
            namespace="default"
        )




import pandas as pd
from langchain_openai import OpenAIEmbeddings
import chromadb
import chromadb.config
import uuid

from llm_engine import get_llm_summary

from dotenv import load_dotenv
load_dotenv()

# 初期化
embedding = OpenAIEmbeddings(model= "text-embedding-3-small")

# 読み込み用データ
df = pd.read_csv("./csv/select50.csv")
texts = df["text"].tolist()
summary_texts = [get_llm_summary(text) for text in texts]
# print(summary_texts)
embeddings = embedding.embed_documents(summary_texts)
titles = df["title"].tolist()
urls = df["url"].tolist()
teachers = df["teacher"].tolist()

# chromadb
persist_directory = "./docs/chroma"
collection_name="langchain_store"

client = chromadb.PersistentClient(path=persist_directory)
collection = client.create_collection(name=collection_name)

collection.add(
    documents = texts,
    embeddings = embeddings,
    metadatas = [{"title": s, "url": l, "teacher": m } for s, l, m in zip(titles, urls, teachers)],
    ids=[str(uuid.uuid1()) for _ in texts]
)
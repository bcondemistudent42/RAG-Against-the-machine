import json

import chromadb


def search(query: str, k: int = 3):  # setting it to 3  for now

    docs = []
    ids = []

    with open("data/processed/my_chunk.json") as json_file:
        data_chunked = json.load(json_file)

    for each_chunk in data_chunked.values():
        docs.append(each_chunk["content"])
        ids.append(str(each_chunk["chunk_idx"]))

    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="my_collection")
    collection.upsert(
        documents=docs, ids=ids
    )
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    print(results)


search("what is vllm")

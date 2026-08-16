




def search(query: str, k: int = 3):  # setting it to 3  for now


# to do in the index part


    # to in the search
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    print(results.keys())


search("who created the program vllm")

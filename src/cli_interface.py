import fire

# my import
from loader import Loader
from my_bm25 import to_Bm25
from indexing import Indexer
from to_json import JsonCreator
from chunker import Chunker, ChunkedData
from ai_answer import Ai_work


import json

# index –max_chunk_size <int>
# Ingest data/raw/ and build the index under data/processed/.


def index(chunk_size: int = 2000):
    database = "data/raw"
    load = Loader(database)
    raw_data = load.load_all()

    my_chunker = Chunker(raw_data, chunk_size)
    chunked_data, metadata_sources = my_chunker.chunk_all()

    my_indexer = Indexer(chunked_data, metadata_sources)
    metadatas = my_indexer.make_all_metadata_index()

    make_json = JsonCreator(chunked_data, metadatas)
    make_json.convert_all()
    make_json.write_chunk()

    bm = to_Bm25(chunked_data)
    bm.convert_to_corpus()
    bm.tokenize_and_index()

def search(query: str, k: int=3): #setting it to 3  for now
    max_relevant = to_Bm25.find_k_relevant_one(query, k)
    with open('data/processed/my_chunk.json') as json_file:
        data_chunked = json.load(json_file)
    cleaned_relevant = max_relevant[0][0]
    print(f"\nTop k = {k} result:")
    for i, chunk_id in enumerate(cleaned_relevant):
        file_path, start_idx, end_idx = Loader.data_from_relevant(
            data_chunked,
            chunk_id
            )
        print(
            f"File rank: {i} File path: {file_path} {start_idx} {end_idx}"
        )
        # to upgrade the output to be better
    print()

def search_dataset(
    dataset_path: str="data/datasets/UnansweredQuestions/dataset_docs_public.json",
    k: int=10,
    save_directory: str="data/output/search_results/"
    ):
    """
    Run search over a whole dataset and write a StudentSearchResults JSON file.
    """
    no_answer_q = Loader.load_questions(dataset_path)
    my_questions = Loader.validate_unanswered_q(no_answer_q)
    max_relevant = to_Bm25.find_k_relevant(my_questions, k)
    output = {}
    output["search_results"] = []
    for i, chunk_id in enumerate(max_relevant):
        k_relevant = chunk_id[0]
        each_q = my_questions[i]
        json_dct = Loader.build_dict(each_q, k_relevant)
        output["search_results"].append(json_dct)
    JsonCreator.write_any_json(save_directory, output)


def answer(query: str, k: int=3):
    my_ai = Ai_work()
    max_relevant = to_Bm25.find_k_relevant_one(query, k)
    answer = my_ai.get_one_answer(query, max_relevant)
    print()
    print(f"Model Answer: {answer}")
    print()


def main():
      fire.Fire({
      'index': index,
      "search": search,
      "search_dataset": search_dataset,
      "answer": answer

  })


# to try improve perf with some np array
if __name__ == '__main__':
    # try:
        main()
    # except FileNotFoundError as e:
        # print("\n===============")
        # print("[ERROR]")
        # print("A required file or folder is missing")
        # print(f"Missing File or Folder: {e.filename}")
        # print("Perhaps you forgot to index ?")
        # print("=================\n")
    # except BaseException as e:
        # print(f"An error occured: {e}")

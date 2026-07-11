from loader import Loader
from indexing import Indexer
from to_json import JsonCreator
from my_bm25 import to_Bm25
from chunker import Chunker, ChunkedData



def extract_questions():
    import json
    with open("datasets_public/public/UnansweredQuestions/dataset_code_public.json") as json_file:
        data = json.load(json_file)
    print(data)

def visualize_chunk(chunked_data: ChunkedData, data_type: str):
    target_data = getattr(chunked_data, data_type)
    for elt in target_data:
        for each in elt:
            print("\n______START_________\n")
            print(each)
            print(f"====== LEN : {len(each)} =======")
            print("\n______END_________\n")

def main():
    database = "vllm-0.10.1" #to define later
    k = 4 # to define later
    # database = "bible"

    load = Loader(database)
    raw_data = load.load_all()

    my_chunker = Chunker(raw_data)
    chunked_data, metadata_sources = my_chunker.chunk_all()

    my_indexer = Indexer(chunked_data, metadata_sources)
    metadatas = my_indexer.make_all_metadata_index()

    make_json = JsonCreator(chunked_data, metadatas)
    make_json.convert_all()
    make_json.write_json()

    bm = to_Bm25(chunked_data, k)
    bm.convert_to_corpus()
    bm.tokenize_and_index()

    extract_questions()



if __name__ == "__main__":
    main()

    # Use BM25 to get the -k most corresponding answers

# > To build dynamically later:
    # chunk size depending on input
    # number of chunk taken depending on K
    # Question to answers
    # Dataset to see if dynamic


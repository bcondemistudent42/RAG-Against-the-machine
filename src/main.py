from loader import Loader
from indexing import Indexer
from to_json import JsonCreator
from my_bm25 import to_Bm25
from chunker import Chunker, ChunkedData

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

    bm = to_Bm25(chunked_data)
    bm.convert_to_corpus()
    bm.tokenize_and_index()



if __name__ == "__main__":
    main()
from loader import Loader
from indexing import Indexer
from chunker import Chunker, ChunkedData
from to_json import JsonCreator


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
    to_json_dump = make_json.convert_all()
    import json
    with open("chunk.txt", "w") as f:
        f.write(json.dumps(to_json_dump, indent=4))


    # print(metadatas.txt[13][0])
    # print("============")
    # print(chunked_data.txt[13][0])
    #                    Files Chunks
    # visualize_chunk(chunked_data, "txt")
    # for elt in clean_meta.txt:
        # for test in elt:
            # print(test.last_character_index - test.first_character_index)


if __name__ == "__main__":
    main()

# Tout transformer en json
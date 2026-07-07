from loader import Loader
from chunker import Chunker, Chunked_data
from indexing import make_index


def visualize_chunk(chunked_data: Chunked_data, data_type: str):
    target_data = getattr(chunked_data, data_type)
    for elt in target_data:
        for each in elt:
            print("\n______START_________\n")
            print(each)
            print(f"====== LEN : {len(each)} =======")
            print("\n______END_________\n")


def main():
    # database = "vllm-0.10.1" #to define later
    database = "bible"
    load = Loader(database)
    raw_data = load.load_all()
    my_chunker = Chunker(raw_data)
    chunked_data, metadata_sources = my_chunker.chunk_all()
    # to do another proper function to call for all file type
    make_index(chunked_data.txt, metadata_sources["txt"])


if __name__ == "__main__":
    main()


# sauvgarder data en json
# dabord regardr comment implementer bm 25
# tout est bien chunker, juste a relier au metadata plus tard

# Clean data structure:
# [list of [chunked_documents]]
# [chunked_document] = [chunk_of_txt] of size 2000
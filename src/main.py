from loader import Loader
from chunker import Chunker


def main():
    database = "vllm-0.10.1" #to define later 
    load = Loader(database)
    raw_data = load.load_all()
    my_chunker = Chunker(raw_data)
    chunked_data = my_chunker.chunk_all()
    print(chunked_data)

if __name__ == "__main__":
    main()
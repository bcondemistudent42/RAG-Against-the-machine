from loader import Loader
from chunker import Chunker


def main():
    # database = "vllm-0.10.1" #to define later
    database = "bible"
    load = Loader(database)
    raw_data = load.load_all()
    my_chunker = Chunker(raw_data)
    chunked_data = my_chunker.chunk_all()
    print((chunked_data.txt))

if __name__ == "__main__":
    main()

    # sauvgarder data en json
    # dabord regardr comment implementer bm 25
    # tout est bien chunker, juste a relier au metadata plus tard
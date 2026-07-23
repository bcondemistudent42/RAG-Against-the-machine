import fire

# my import
from loader import Loader
from my_bm25 import to_Bm25
from indexing import Indexer
from to_json import JsonCreator
from chunker import Chunker, ChunkedData
from ai_answer import Ai_work


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
    make_json.write_json()

    bm = to_Bm25(chunked_data)
    bm.convert_to_corpus()
    bm.tokenize_and_index()

if __name__ == '__main__':
  fire.Fire({
      'index': index,
  })
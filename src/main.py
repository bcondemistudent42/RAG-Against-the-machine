from loader import Loader
from my_bm25 import to_Bm25
from indexing import Indexer
from to_json import JsonCreator
from chunker import Chunker, ChunkedData
from ai_answer import Ai_work


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
    k = 1 # to define later
    path_of_questions = "datasets_public/public/UnansweredQuestions/dataset_docs_public.json"
    # to define later

    load = Loader(database)
    raw_data = load.load_all()
    no_answer_q = load.load_questions(path_of_questions)
    questions = load.validate_unanswered_q(no_answer_q)

    my_chunker = Chunker(raw_data)
    chunked_data, metadata_sources = my_chunker.chunk_all()

    my_indexer = Indexer(chunked_data, metadata_sources)
    metadatas = my_indexer.make_all_metadata_index()

    make_json = JsonCreator(chunked_data, metadatas)
    make_json.convert_all()
    make_json.write_json()

    bm = to_Bm25(chunked_data)
    bm.convert_to_corpus()
    bm.tokenize_and_index() #to execute only for index
    max_relevant = bm.find_k_relevant(questions, k)

    my_ai = Ai_work()
    my_ai.get_answers(questions, max_relevant)



if __name__ == "__main__":
    # try:
        main()
    # except Exception as e:
        # print("[ERROR]: ", e)

# > To build dynamically later:
    # chunk size depending on input
    # number of chunk taken depending on K
    # Question to answers
    # Dataset to see if dynamic
    # to do the cli with python fire
    # to do chroma db abd embedding

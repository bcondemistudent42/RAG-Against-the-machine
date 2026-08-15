import json

import fire
from pydantic import ValidationError
from tqdm import tqdm

from .ai_answer import Ai_work
from .chunker import Chunker
from .indexing import Indexer
from .loader import Loader
from .my_bm25 import to_Bm25
from .required_class import ChunksLst
from .to_json import JsonCreator


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


def search(query: str, k: int = 3):  # setting it to 3  for now

    if k < 1 or k > 10:
        raise ValueError(f"K have to be in the range 1 <= k <= 10\nActual k={k}")
    max_relevant = to_Bm25.find_k_relevant_one(query, k)
    with open("data/processed/my_chunk.json") as json_file:
        data_chunked = json.load(json_file)
        ChunksLst.validate(data_chunked)

    cleaned_relevant = max_relevant[0][0] 
    iterable = cleaned_relevant
    output = ""

    with tqdm(total=len(iterable)) as pbar:
        for i, chunk_id in enumerate(iterable):
            fp, s_idx, end_idx = Loader.data_from_relevant(
                data_chunked, chunk_id
            )
            output += f"File rank: {i + 1} File path: {fp} {s_idx} {end_idx}\n"
            pbar.update(1)
    print(f"\nTop k = {k} result:")
    print(output)
    # to upgrade the output to be better


def search_dataset(
    dataset_path: str
    = "data/datasets/UnansweredQuestions/dataset_code_public.json",
    k: int = 5,
    save_directory: str = "data/output/search_results/",
):
    """
    Run search over a whole dataset and write a StudentSearchResults JSON file.
    """

    if k < 1 or k > 10:
        raise ValueError(f"K have to be in the range 1 <= k <= 10\nActual k={k}")

    no_answer_q = Loader.load_questions(dataset_path)
    my_questions = Loader.validate_unanswered_q(no_answer_q)

    max_relevant = to_Bm25.find_k_relevant(my_questions, k=k)

    output = {}
    output["search_results"] = []
    output["k"] = k

    iterable = max_relevant

    with tqdm(total=len(iterable)) as pbar:
        for i, chunk_id in enumerate(iterable):
            k_relevant = chunk_id[0]
            each_q = my_questions[i]
            json_dct = Loader.build_dict(each_q, k_relevant)
            output["search_results"].append(json_dct)
            pbar.update(1)

    JsonCreator.write_any_json(save_directory, output)
    print()
    print(f"Saved results at {save_directory}")
    print()


def answer(query: str, k: int = 5):

    if k < 1 or k > 10:
        raise ValueError(f"K have to be in the range 1 <= k <= 10\nActual k={k}")

    my_ai = Ai_work()
    # maybe to open before the search result. to see if relevant
    max_relevant = to_Bm25.find_k_relevant_one(query, k)
    answer = my_ai.get_one_answer(query, max_relevant)
    print()
    print(f"Model Answer: {answer}")
    print()
    # improve output


def answer_dataset(
    student_search_results_path: str
    = "data/output/search_results/search_results.json",
    save_directory: str = "data/output/search_results_and_answer",
):
# to protect all input from function call
    path_of_questions = (
        "data/datasets/UnansweredQuestions/dataset_docs_public.json"
    )
    no_answer_q = Loader.load_questions(path_of_questions)
    questions = Loader.validate_unanswered_q(no_answer_q)

    my_ai = Ai_work()
    # to do with the precedent result saved at
    # data/output/search_results/UnansweredQuestions/dataset_docs_public.json

    with open(student_search_results_path) as json_file:
        sources = json.load(json_file)
        # to check if it's a correct StudentSearchResults
        # with baseModel
    answers = my_ai.get_answers(sources)

    # formatting part
    output = {}
    output["search_results"] = []
    output["k"] = sources["k"]

    # to tqdm later can't test now
    # to test at home if quick or no
    for i, each_question in enumerate(questions):
        little_dct = Loader.build_dict_answer(
            each_question,
            sources["search_results"][i]["retrieved_sources"],
            answers[i],
        )
        output["search_results"].append(little_dct)

    JsonCreator.write_any_json(
        save_directory, output, "search_results_and_answer.json"
    )
    print(f"\nResult were saved at '{save_directory}'\n")


def main():
    fire.Fire(
        {
            "index": index,
            "search": search,
            "search_dataset": search_dataset,
            "answer": answer,
            "answer_dataset": answer_dataset,
        }
    )


# to do the recallok stuff to see later
# to do the evaluate stuff dont know how it works yet
# before taking any data fron json check structure with pydantic
# to implement the rff ranking with chomadb
# to do the cache upgrade

# to try improve perf with some np array
if __name__ == "__main__":
    try:
        main()
    except json.JSONDecodeError as e:
        print("\n===============")
        print("[ERROR]")
        print(f"JSON DECODE ERROR OCURED :\n{e}")
        print("=================\n")
    except FileNotFoundError as e:
        print("\n===============")
        print("[ERROR]")
        print("A required file or folder is missing")
        print(f"Missing File or Folder: {e.filename}")
        print("Perhaps you forgot to index ?")
        print("=================\n")
    except ValidationError as e:
        print("\n===============")
        print("[PYDANTIC VALIDATION ERROR] :")
        print("Hint from Pydantic")
        print(f"{e.errors()[0]['msg']}")
        print(f"{e.errors()[0]['type']}")
        print(f"{e.errors()[0]['loc']}")
        print("=================\n")
    except ValueError as e:
        print("\n===============")
        print("[ERROR]")
        print(f"A given value is missing or incorrect: {e}")
        print("=================\n")
    except BaseException as e:
        print(f"An error occured: {e}")



        # for error in e.errors():
            # field_name = error['loc'][-1]
            # msg = error['msg'].replace("Value error, ", "")
            # print(f"Error on field '{field_name}': {msg}")
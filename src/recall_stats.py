import json

class Recall:

    def __init__(self, search_rslt: str, dataset_path: str):
        self.search_rslt = search_rslt
        self.dataset_path = dataset_path

    def evaluate(self):
        ...

    def check_answer(self):
        with open(self.search_rslt) as json_file:
            result = json.load(json_file)
        with open(self.dataset_path) as json_file:
            answered_q = json.load(json_file)

        l_search_result = len(result["search_results"])
        rag_q = answered_q["rag_questions"]
        rslt_q = result["search_results"]

        if (l_search_result != len(answered_q["rag_questions"])):
            print("\n[Error]: SearchResult must have the same number of AnsweredQuestion\n")

        for i in range(l_search_result):
            if rslt_q[i]["question_id"] != rag_q[i]["question_id"]:
                print("[Error] Not the same question id, maybe wrong datset answered")
                print(f"{rslt_q[i]['question_id']} vs {rag_q[i]['question_id']}")




test = Recall(
    "data/output/search_results/search_results.json",
    "data/datasets/AnsweredQuestions/dataset_docs_public.json")
test.check_answer()
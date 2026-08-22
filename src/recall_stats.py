import json
import sys


class Recall:

    def __init__(self, search_rslt: str, dataset_path: str):
        with open(search_rslt) as json_file:
            self.result = json.load(json_file)
        with open(dataset_path) as json_file:
            self.answered_q = json.load(json_file)

    def workout_overlap(self, correct: tuple, to_test: tuple):
        if correct[0] >= correct[1] or to_test[0] >= to_test[1]:
            return False

        intersection = max(0, min(correct[1], to_test[1]) -
                           max(correct[0], to_test[0]))
        union = max(correct[1], to_test[1]) - min(correct[0], to_test[0])

        return True if union > 0 and intersection / union >= 0.05 else False

    def evaluate(self, k: int):
        score = 0
        rag_source = self.answered_q["rag_questions"]
        result_source = self.result["search_results"]
        for i in range(len(rag_source)):

            file_path_correct = rag_source[i]["sources"][0]["file_path"]
            first_correct = rag_source[i]["sources"][0][
                "first_character_index"
            ]
            last_correct = rag_source[i]["sources"][0][
                "last_character_index"
            ]

            file_path_answer = result_source[i]["retrieved_sources"]
            for j, each_file in enumerate(file_path_answer):
                if j >= k:
                    break
                if each_file["file_path"] == file_path_correct:
                    start_idx_check = each_file["first_character_index"]
                    end_idx_check = each_file["last_character_index"]
                    if self.workout_overlap(
                        (first_correct, last_correct),
                        (start_idx_check, end_idx_check)
                    ):
                        score += 1
                        break
        return score / 100

    def check_answer(self):

        l_search_result = len(self.result["search_results"])
        rag_q = self.answered_q["rag_questions"]
        rslt_q = self.result["search_results"]

        if (l_search_result != len(self.answered_q["rag_questions"])):
            print(l_search_result)
            print(len(self.answered_q["rag_questions"]))
            print(
                "\n[Error]: SearchResult must have",
                "the same number of AnsweredQuestion\n"
            )
            sys.exit(1)

        self.lentgh = l_search_result
        for i in range(l_search_result):
            if rslt_q[i]["question_id"] != rag_q[i]["question_id"]:
                print(
                    "[Error] Not the same question id,",
                    "maybe wrong dataset answered"
                )
                print(
                    f"{rslt_q[i]['question_id']} vs {rag_q[i]['question_id']}"
                )
                sys.exit(1)

    def evaluate_all(self):
        recall = [1, 3, 5, 10]

        print("Student data is valid: True")
        print(f"Total number of questions: {self.lentgh}")
        print("\nEvaluation Result")
        print("=============")

        for each_recall in recall:
            score = self.evaluate(each_recall)
            print(f"Recall@{each_recall}: {score} ({score * 100}%)")

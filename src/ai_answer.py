import json

import dspy
from tqdm import tqdm


# faire une autre classe qui recois une liste de question
# elle traite ensuite tout les questions une par une et stock
# le resultat dans un json similaire a asnwered questions
class AnswerBot(dspy.Signature):
    """
    Answer the question with the given data.
    """
    question: str = dspy.InputField(desc="Question to answer")
    data: str = dspy.InputField(desc="Data linked to the question.")
    answer: str = dspy.OutputField()


class Ai_work:
    def __init__(self):
        with open('data/processed/my_chunk.json') as json_file:
            data_chunked = json.load(json_file)
        self.data_chunked = data_chunked
        self.lm = dspy.LM(
            "openai/Qwen/Qwen3-0.6B",
            api_base="http://localhost:8000/v1",
            api_key="_"
        )
        dspy.configure(lm=self.lm)

    def get_answers(self, index_of_k):
        output = []
        # to do later can't test now tqdm
        reasoning_bot = dspy.ChainOfThought(AnswerBot)

        iterable = index_of_k["search_results"]

        with tqdm(total=len(iterable)) as pbar:
            for each_question in iterable:
                for each_source in each_question["retrieved_sources"]:
                    sources_data = ""
                    with open(each_source["file_path"]) as f:
                        test = f.read()
                        start_index = each_source["first_character_index"]
                        last_index = each_source["last_character_index"]
                        sources_data += f"\n {test[start_index:last_index]}"
                result = reasoning_bot(
                    data=sources_data,
                    question=each_question["question"]
                )
                pbar.update(1)
            # print("Question :", each_question["question"])
            # print("Answer :", result.answer)
            output.append(result.answer)
        return output

    def get_one_answer(self, question: str, index_of_k):
        reasoning_bot = dspy.ChainOfThought(AnswerBot)
        result = reasoning_bot(
            data=self.data_chunked[str(index_of_k[0][0][0])]["content"],
            question=question
        )
        return (result.answer)

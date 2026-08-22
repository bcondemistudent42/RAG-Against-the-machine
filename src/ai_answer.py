import json
from typing import Any

import dspy
from tqdm import tqdm


# faire une autre classe qui recois une liste de question
# elle traite ensuite tout les questions une par une et stock
# le resultat dans un json similaire a asnwered questions
class AnswerBot(dspy.Signature):
    """Answer a question using the provided context data."""
    question: str = dspy.InputField(desc="Question to answer")
    data: str = dspy.InputField(desc="Data linked to the question.")
    answer: str = dspy.OutputField()


class Ai_work:
    def __init__(self) -> None:
        """Initialize the answer-generation pipeline."""
        with open('data/processed/my_chunk.json') as json_file:
            data_chunked = json.load(json_file)
        self.data_chunked = data_chunked
        self.lm = dspy.LM(
            "openai/Qwen/Qwen3-0.6B",
            api_base="http://localhost:8000/v1",
            api_key="_"
        )
        dspy.configure(lm=self.lm)

    def get_answers(self, index_of_k: dict[str, Any]) -> list[str]:
        """Generate answers for a batch of search results.

        Args:
            index_of_k: Search-result payload containing retrieved sources.

        Returns:
            A list of generated answers in question order.
        """
        output = []
        reasoning_bot = dspy.ChainOfThought(AnswerBot)

        iterable = index_of_k["search_results"]

        with tqdm(
            total=len(iterable), desc="Generating Answer with VLLM"
        ) as pbar:
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
                output.append(result.answer)
        return output

    def get_one_answer(
        self,
        question: str,
        index_of_k: list[list[list[int]]],
    ) -> str:
        """Generate one answer from the top retrieved chunk.

        Args:
            question: Question to answer.
            index_of_k: Retrieved chunk ranking data.

        Returns:
            The generated answer text.
        """
        reasoning_bot = dspy.ChainOfThought(AnswerBot)
        result = reasoning_bot(
            data=self.data_chunked[str(index_of_k[0][0][0])]["content"],
            question=question
        )
        return str(result.answer)

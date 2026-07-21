import dspy
import json
from required_class import UnansweredQuestion

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
        with open('chunk.json') as json_file:
            data_chunked = json.load(json_file)
        self.data_chunked = data_chunked
        self.lm = dspy.LM(
            "openai/Qwen/Qwen3-0.6B",
            api_base="http://localhost:8000/v1",
            api_key="_"
        )
        dspy.configure(lm=self.lm)

    def get_answers(self, questions: list[UnansweredQuestion], index_of_k):
        reasoning_bot = dspy.ChainOfThought(AnswerBot)
        for i, each_question in enumerate(questions):
            # print() #to see what format maybe to send all content with a get_content
            result = reasoning_bot(
            data=self.data_chunked[str(index_of_k[i][0][0])]["content"],
            question=each_question.question
            )
            print("\n=========")
            print(result.answer)
            print("=========\n")

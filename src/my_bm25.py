import bm25s
from required_class import UnansweredQuestion
from my_enum import FileType


class to_Bm25:
    def __init__(self, chunked_data):
        self.chunked_data = chunked_data
        self.corpus = []

    def convert_to_corpus(self):
        for data_type in FileType:
            for each_file in getattr(self.chunked_data, data_type):
                for each_chunk in each_file:
                    self.corpus.append(each_chunk)

    def tokenize_and_index(self):
        if len(self.corpus) == 0:
            raise ValueError("No data was given")
        corpus_tokens = bm25s.tokenize(self.corpus)
        retriever = bm25s.BM25(corpus=self.corpus)
        retriever.index(corpus_tokens)
        retriever.save("data/processed/bm25_index")
        self.retriever = retriever

    @staticmethod
    def find_k_relevant(questions: list[UnansweredQuestion], k: int):
        search_k_retreiver = bm25s.BM25.load("data/processed/bm25_index")
        output = []
        for each_question in questions:
            query_tokens = bm25s.tokenize([each_question.question])
            docs, _ = search_k_retreiver.retrieve(query_tokens, k=k)
            output.append(docs.tolist())
        return (output)

    @staticmethod
    def find_k_relevant_one(question: UnansweredQuestion, k: int):
        search_k_retreiver = bm25s.BM25.load("data/processed/bm25_index")
        output = []
        query_tokens = bm25s.tokenize([question])
        docs, _ = search_k_retreiver.retrieve(query_tokens, k=k)
        output.append(docs.tolist())
        return (output)
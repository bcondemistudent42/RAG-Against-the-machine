import bm25s
from required_class import UnansweredQuestion
from my_enum import FileType


class to_Bm25:
    def __init__(self, chunked_data, k: int):
        self.chunked_data = chunked_data
        self.corpus = []
        self.k = k

    def convert_to_corpus(self):
        for data_type in FileType:
            for each_file in getattr(self.chunked_data, data_type):
                for each_chunk in each_file:
                    self.corpus.append(each_chunk)

    def tokenize_and_index(self):
        corpus_tokens = bm25s.tokenize(self.corpus)
        retriever = bm25s.BM25(corpus=self.corpus)
        retriever.index(corpus_tokens)
        retriever.save("indexing")
        self.retriever = retriever

    def find_k_relevant(self, questions: list[UnansweredQuestion]):
        search_k_retreiver = bm25s.BM25.load("indexing")
        output = []
        for each_question in questions:
            query_tokens = bm25s.tokenize([each_question.question])
            docs, _ = search_k_retreiver.retrieve(query_tokens, k=self.k)
            output.append(docs.tolist())
        return (output)
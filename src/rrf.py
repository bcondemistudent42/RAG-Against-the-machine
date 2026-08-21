from abc import ABC, abstractmethod
from collections import OrderedDict


class Rrf(ABC):

    @staticmethod
    def get_score(rank: int):
        ranking_constant = 60
        return 1.0 / (ranking_constant + rank)

    @abstractmethod
    def output_rff(self):
        ...


class Rrf_simple_search(Rrf):
    def __init__(self, bm25_results: list[int], embedding_results: list[int]):
        self.bm_rslt = bm25_results
        self.chroma_rslt = [int(x) for x in embedding_results]

    def output_rff(self):
        scores = {}
        for i, elt in enumerate(self.bm_rslt):
            if elt not in scores:
                scores[elt] = Rrf.get_score(i + 1)
            else:
                scores[elt] += Rrf.get_score(i + 1)
        for i, elt in enumerate(self.chroma_rslt):
            if elt not in scores:
                scores[elt] = Rrf.get_score(i + 1)
            else:
                scores[elt] += Rrf.get_score(i + 1)

        output = [
            x[0] for x in sorted(scores.items(),
            key=lambda x: x[1],
            reverse=True)
            ]
        return list(OrderedDict.fromkeys(output))



class Rrf_dataset_search(Rrf):
    def __init__(self, bm25_results, embedding_results, k):
        self.bm_rslt = [x[0] for x in bm25_results]
        self.chroma_rslt = []
        for each_list in embedding_results:
            little_lst = []
            for each_number in each_list:
                little_lst.append(int(each_number))
            self.chroma_rslt.append(little_lst)
        self.k = k

    def output_rff(self):
        final = []
        for j in range(len(self.chroma_rslt)):
            scores = {}
            for i in range(len(self.bm_rslt[j])):
                if self.bm_rslt[j][i] not in scores:
                    scores[self.bm_rslt[j][i]] = Rrf.get_score(i + 1)
                else:
                    scores[self.bm_rslt[j][i]] += Rrf.get_score(i + 1)
            for i in range(len(self.chroma_rslt[j])):
                if self.chroma_rslt[j][i] not in scores:
                    scores[self.chroma_rslt[j][i]] = Rrf.get_score(i + 1)
                else:
                    scores[self.chroma_rslt[j][i]] += Rrf.get_score(i + 1)
            output = [
                x[0] for x in sorted(scores.items(),
                key=lambda x: x[1],
                reverse=True)
                ]
            output = list(OrderedDict.fromkeys(output))
            final.append(output[0:self.k])
        return final


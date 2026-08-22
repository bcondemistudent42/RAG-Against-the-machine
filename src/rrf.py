from abc import ABC, abstractmethod
from collections import OrderedDict


class Rrf(ABC):

    @staticmethod
    def get_score(rank: int):
        """Return the reciprocal-rank fusion score for one position.

        Args:
            rank: One-based rank in the input list.

        Returns:
            The reciprocal-rank fusion contribution for the rank.
        """
        ranking_constant = 60
        return 1.0 / (ranking_constant + rank)

    @abstractmethod
    def output_rff(self):
        """Return the fused ranking output."""
        ...


class Rrf_simple_search(Rrf):
    def __init__(self, bm25_results: list[int], embedding_results: list[int]):
        """Initialize reciprocal-rank fusion for a single query.

        Args:
            bm25_results: BM25 ranking results.
            embedding_results: Embedding ranking results.
        """
        self.bm_rslt = bm25_results
        self.chroma_rslt = [int(x) for x in embedding_results]

    def output_rff(self):
        """Fuse BM25 and embedding rankings for one query.

        Returns:
            A de-duplicated list of fused chunk identifiers.
        """
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

        output = [x[0] for x in sorted(scores.items(), key=lambda x: x[1],
                  reverse=True)]
        return list(OrderedDict.fromkeys(output))


class Rrf_dataset_search(Rrf):
    def __init__(self, bm25_results, embedding_results, k):
        """Initialize reciprocal-rank fusion for a dataset.

        Args:
            bm25_results: BM25 ranking results per question.
            embedding_results: Embedding ranking results per question.
            k: Number of fused results to keep per question.
        """
        self.bm_rslt = [x[0] for x in bm25_results]
        self.chroma_rslt = []
        for each_list in embedding_results:
            little_lst = []
            for each_number in each_list:
                little_lst.append(int(each_number))
            self.chroma_rslt.append(little_lst)
        self.k = k

    def output_rff(self):
        """Fuse rankings for all questions in the dataset.

        Returns:
            A list of top-k fused chunk identifiers per question.
        """
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
            output = [x[0] for x in sorted(scores.items(),
                      key=lambda x: x[1], reverse=True)]
            output = list(OrderedDict.fromkeys(output))
            final.append(output[0:self.k])
        return final

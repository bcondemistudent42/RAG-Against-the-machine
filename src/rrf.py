class Rrf:
    def __init__(self, bm25_results: list[int], embedding_results: list[int]):
        self.bm_rslt = bm25_results
        self.chroma_rslt = embedding_results

    @staticmethod
    def get_score(rank: int):
        ranking_constant = 60
        return 1.0 / (ranking_constant + rank)

    def output_rff(self):
        scores = {}
        for i, elt in enumerate(self.bm_rslt):
            scores[elt] = self.get_score(i + 1)
        for i, elt in enumerate(self.chroma_rslt):
            scores[elt] = self.get_score(i + 1)
        output = [
            x[0] for x in sorted(scores.items(),
            key=lambda x: x[1],
            reverse=True)
            ]
        return output

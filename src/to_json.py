import json
from dataclasses import asdict
from my_enum import FileType
from chunker import ChunkedData
from indexing import OrganisedMetadata, MinimalSource


class JsonCreator:
    def __init__(self, chunks: ChunkedData, metadata: OrganisedMetadata):
        self.chunks = chunks
        self.metadata = metadata

    def convertor(self):
        chunk_idx = 0
        big_lst = []
        for i in range(len(self.chunks.txt)):
            for j in range(len(self.chunks.txt[i])):
                temp = {
                    "chunk_idx": chunk_idx,
                    "file_path": self.metadata.txt[i][j].file_path,
                    "first_character_index": self.metadata.txt[i][j].first_character_index,
                    "last_character_index": self.metadata.txt[i][j].last_character_index,
                    "content": self.chunks.txt[i][j]
                }
                big_lst.append(temp)
                chunk_idx += 1
        return (big_lst)




# chunk id for all chunk not unique for each file





# file_path: str
# first_character_index: int
# last_character_index: int
# index
# chunk
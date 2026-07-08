import json
from dataclasses import asdict
from my_enum import FileType
from chunker import ChunkedData
from indexing import OrganisedMetadata, MinimalSource


class JsonCreator:
    def __init__(self, chunks: ChunkedData, metadata: OrganisedMetadata):
        self.big_dict = {}
        self.chunks = chunks
        self.metadata = metadata

    def _convertor(self, data_type: FileType, chunk_idx: int):
        for i in range(len(getattr(self.chunks, data_type))):
            for j in range(len(getattr(self.chunks, data_type)[i])):
                temp = {
                    "chunk_idx": chunk_idx,
                    "file_path": getattr(self.metadata, data_type)[i][j].file_path,
                    "first_character_index": getattr(self.metadata, data_type)[i][j].first_character_index,
                    "last_character_index": getattr(self.metadata, data_type)[i][j].last_character_index,
                    "content": getattr(self.chunks, data_type)[i][j]
                }
                self.big_dict[chunk_idx] = temp
                chunk_idx += 1

    def convert_all(self):
        chunk_idx = 0
        for data_type in FileType:
            self._convertor(data_type, chunk_idx)
            chunk_idx = len(self.big_dict.keys())
        return self.big_dict



# chunk id for all chunk not unique for each file





# file_path: str
# first_character_index: int
# last_character_index: int
# index
# chunk
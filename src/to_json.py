import json
from .my_enum import FileType
from .chunker import ChunkedData
from .indexing import OrganisedMetadata


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

# to secure if processed folder not created
    def write_chunk(self):
        with open("data/processed/my_chunk.json", "w") as f:
            f.write(json.dumps(self.big_dict, indent=4))

    @staticmethod
    def write_any_json(directory_to_write: str, the_dict: dict, the_name="search_results.json"):
        to_write = f"{directory_to_write}/{the_name}"
        # to see how to handle when folder not exist
        # to see with gemini
        try:
            with open(to_write, "w") as f:
                f.write(json.dumps(the_dict, indent=4))
        except Exception as e:
            raise ValueError(
                f'Error occurred while saving JSON to {directory_to_write}: {e}'
            )
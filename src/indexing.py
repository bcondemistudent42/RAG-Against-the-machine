from my_enum import FileType
from pydantic import BaseModel
from chunker import Chunked_data
from dataclasses import dataclass

class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

@dataclass
class Organised_Metadata:
    py: list[MinimalSource]
    md: list[MinimalSource]
    txt: list[MinimalSource]

class Indexer:
    def __init__(self, chunked_data: Chunked_data, all_sources: dict[str : str]):
        self.chunked_data = chunked_data
        self.all_sources = all_sources

    def make_all_metadata_index(self):
        output = {}
        for data_type in FileType:
            data_type = str(data_type)
            output[data_type] = []
            for each_file in getattr(self.chunked_data, data_type):
                temp = self._make_metadata_index(each_file, self.all_sources[data_type])
                output[data_type].append(temp)
        return (Organised_Metadata(**output))

    @staticmethod
    def _make_metadata_index(typed_data: list[list[str]], metadata_typed: list[str]) -> list[MinimalSource]:
        output = []
        prev_len = 0
        for chunk in typed_data:
            actual_len = len(chunk)
            output.append(
                MinimalSource(
                file_path=metadata_typed[0][0],
                first_character_index=prev_len,
                last_character_index=prev_len + actual_len
                )
            )
            prev_len += actual_len + 1
        return output
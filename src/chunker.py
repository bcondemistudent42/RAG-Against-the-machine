from my_enum import FileType
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Clean_data:
    py: list[str]
    md: list[str]
    txt: list[str]


class Chunker():
    def __init__(self, raw_data):
        self.raw_data = raw_data

    @staticmethod
    def splitter(typed_data: Document, type: FileType):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        chunk = text_splitter.split_text(str(typed_data))
        #to filter depending on type
        # for elt in chunks:
            # print("\n_____START CHUNK______")
            # print(f"{elt}")
            # print("_____END CHUNK______\n")
        return chunk

    def chunk_all(self) -> Clean_data:
        output = {}
        for each_type in FileType:
            each_type = str(each_type)
            output[each_type] = self.splitter(getattr(self.raw_data, each_type), each_type)
        return Clean_data(**output)


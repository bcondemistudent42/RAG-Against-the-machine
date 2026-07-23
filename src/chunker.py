from loader import Raw_data
from my_enum import FileType
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


@dataclass
class ChunkedData:
    py: list[Document]
    md: list[Document]
    txt: list[Document]


class Chunker():
    def __init__(self, raw_data: Raw_data, chunk_size: int):
        self.raw_data = raw_data
        self.chunk_size = chunk_size

    def splitter(self, typed_data: Document, doc_type: FileType):
        if doc_type == FileType.PY:
            multi_splitter = RecursiveCharacterTextSplitter.from_language(
                chunk_size=self.chunk_size,
                chunk_overlap=0, #setting it to 0 to see later
                language=Language.PYTHON
            )
        elif doc_type == FileType.MD:
            multi_splitter = RecursiveCharacterTextSplitter.from_language(
                chunk_size=self.chunk_size,
                chunk_overlap=0, #setting it to 0 to see later
                language=Language.MARKDOWN
            )
        else:
            multi_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=0 #setting it to 0 to see later
            )

        chunk = [multi_splitter.split_text(x.page_content) for x in typed_data]
        metadata_srcs = [x.metadata["source"] for x in typed_data]
        return (chunk, metadata_srcs)

    def chunk_all(self) -> ChunkedData:
        output = {}
        metadata_srcs = {}
        for each_type in FileType:
            each_type = str(each_type)
            output[each_type], temp_meta = self.splitter(getattr(self.raw_data, each_type), each_type)
            metadata_srcs[each_type] = []
            metadata_srcs[each_type].append(temp_meta)
        return (ChunkedData(**output), metadata_srcs)
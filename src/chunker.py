from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from .loader import Raw_data
from .my_enum import FileType


@dataclass
class ChunkedData:
    py: list[list[str]]
    md: list[list[str]]
    txt: list[list[str]]


class Chunker:
    def __init__(self, raw_data: Raw_data, chunk_size: int) -> None:
        """Initialize the chunker.

        Args:
            raw_data: Loaded raw documents grouped by file type.
            chunk_size: Maximum size of each generated chunk.
        """
        self.raw_data = raw_data
        self.chunk_size = chunk_size

    def splitter(
        self,
        typed_data: list[Document],
        doc_type: FileType,
    ) -> tuple[list[list[str]], list[str]]:
        """Split documents of one type into chunks.

        Args:
            typed_data: Documents to split.
            doc_type: File type controlling the splitter configuration.

        Returns:
            A tuple of chunk lists and their source paths.
        """
        if doc_type == FileType.PY:
            multi_splitter = RecursiveCharacterTextSplitter.from_language(
                chunk_size=self.chunk_size,
                chunk_overlap=0,  # setting it to 0 to see later
                language=Language.PYTHON
            )
        elif doc_type == FileType.MD:
            multi_splitter = RecursiveCharacterTextSplitter.from_language(
                chunk_size=self.chunk_size,
                chunk_overlap=0,  # setting it to 0 to see later
                language=Language.MARKDOWN
            )
        else:
            multi_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=0  # setting it to 0 to see later
            )

        chunk = [multi_splitter.split_text(x.page_content) for x in typed_data]
        metadata_srcs = [x.metadata["source"] for x in typed_data]
        return (chunk, metadata_srcs)

    def chunk_all(self) -> tuple[ChunkedData, dict[str, list[list[str]]]]:
        """Chunk all loaded documents by file type.

        Returns:
            Chunked documents and their grouped metadata sources.
        """
        output: dict[str, list[list[str]]] = {}
        metadata_srcs: dict[str, list[list[str]]] = {}
        for each_type in FileType:
            each_type = str(each_type)
            output[each_type], temp_meta = self.splitter(
                getattr(self.raw_data, each_type), each_type
            )
            metadata_srcs[each_type] = []
            metadata_srcs[each_type].append(temp_meta)
        return (ChunkedData(**output), metadata_srcs)

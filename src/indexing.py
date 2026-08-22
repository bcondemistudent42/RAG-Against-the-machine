import json
from dataclasses import dataclass

import chromadb
from pydantic import BaseModel
from tqdm import tqdm

from .chunker import ChunkedData
from .my_enum import FileType


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


@dataclass
class OrganisedMetadata:
    py: list[list[MinimalSource]]
    md: list[list[MinimalSource]]
    txt: list[list[MinimalSource]]


class Indexer:
    def __init__(
        self,
        chunked_data: ChunkedData,
        all_sources: dict[str, list[list[str]]]
    ):
        """Initialize the indexer.

        Args:
            chunked_data: Chunked documents grouped by file type.
            all_sources: Metadata source paths for each chunk group.
        """
        self.chunked_data = chunked_data
        self.all_sources = all_sources

    def make_all_metadata_index(self) -> OrganisedMetadata:
        """Build metadata indices for all chunked file types.

        Returns:
            An `OrganisedMetadata` container with per-chunk metadata.
        """
        output: dict[str, list[list[MinimalSource]]] = {}
        for data_type in FileType:
            data_type = str(data_type)
            output[data_type] = []
            temp = self._make_metadata_index(
                getattr(self.chunked_data, data_type),
                self.all_sources[data_type]
            )
            output[data_type] = temp
        return (OrganisedMetadata(**output))

    @staticmethod
    def _make_metadata_index(
        typed_data: list[list[str]],
        metadata_typed: list[list[str]],
    ) -> list[list[MinimalSource]]:
        """Create metadata entries for a single file type.

        Args:
            typed_data: Nested chunk contents for one file type.
            metadata_typed: Source file paths associated with the chunks.

        Returns:
            Nested metadata objects aligned with the chunk contents.
        """
        file = -1
        output: list[list[MinimalSource]] = []
        for chunk in typed_data:
            file += 1
            prev_len = 0
            all_file_metadata = []
            for each_file in chunk:
                actual_len = len(each_file)
                all_file_metadata.append(
                    MinimalSource(
                        file_path=metadata_typed[0][file],
                        first_character_index=prev_len,
                        last_character_index=prev_len + actual_len
                    )
                )
                prev_len += actual_len + 1
            output.append(all_file_metadata)
        return output

    @staticmethod
    def my_split(arr: list[str], size: int) -> list[list[str]]:
        """Split a list into fixed-size batches.

        Args:
            arr: List to split.
            size: Maximum batch size.

        Returns:
            A list of list batches.
        """
        arrs: list[list[str]] = []
        while len(arr) > size:
            pice = arr[:size]
            arrs.append(pice)
            arr = arr[size:]
        arrs.append(arr)
        return arrs

    @staticmethod
    def embedding() -> None:
        """Embed chunk contents into the persistent Chroma collection."""
        docs = []
        ids = []

        with open("data/processed/my_chunk.json") as json_file:
            data_chunked = json.load(json_file)

        for each_chunk in data_chunked.values():
            docs.append(each_chunk["content"])
            ids.append(str(each_chunk["chunk_idx"]))

        docs_clean = Indexer.my_split(docs, 128)
        ids_clean = Indexer.my_split(ids, 128)
        chroma_client = chromadb.PersistentClient(path="chroma_cache")
        collection = chroma_client.get_or_create_collection(
            name="my_collection"
        )

        for i in tqdm(range(len(docs_clean)), desc="Embedding Chroma DB"):
            collection.add(
                documents=docs_clean[i], ids=ids_clean[i]
            )

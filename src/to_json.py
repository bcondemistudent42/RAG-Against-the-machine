import json

from .chunker import ChunkedData
from .indexing import OrganisedMetadata
from .my_enum import FileType


class JsonCreator:
    def __init__(self, chunks: ChunkedData, metadata: OrganisedMetadata):
        """Initialize the JSON exporter.

        Args:
            chunks: Chunked content grouped by file type.
            metadata: Per-chunk metadata aligned with the content.
        """
        self.big_dict = {}
        self.chunks = chunks
        self.metadata = metadata

    def _convertor(self, data_type: FileType, chunk_idx: int):
        """Convert one file type into the internal JSON structure.

        Args:
            data_type: File type being exported.
            chunk_idx: Starting chunk identifier.
        """
        for i in range(len(getattr(self.chunks, data_type))):
            for j in range(len(getattr(self.chunks, data_type)[i])):
                temp = {
                    "chunk_idx": chunk_idx,
                    "file_path": getattr(
                        self.metadata, data_type
                        )[i][j].file_path,
                    "first_character_index": getattr(
                        self.metadata, data_type
                        )[i][j].first_character_index,
                    "last_character_index": getattr(
                        self.metadata, data_type
                        )[i][j].last_character_index,
                    "content": getattr(self.chunks, data_type)[i][j]
                }
                self.big_dict[chunk_idx] = temp
                chunk_idx += 1

    def convert_all(self):
        """Convert every chunk into the export dictionary.

        Returns:
            The complete chunk mapping ready for serialization.
        """
        chunk_idx = 0
        for data_type in FileType:
            self._convertor(data_type, chunk_idx)
            chunk_idx = len(self.big_dict.keys())
        return self.big_dict

# to secure if processed folder not created
    def write_chunk(self):
        """Write the processed chunk mapping to disk."""
        with open("data/processed/my_chunk.json", "w") as f:
            f.write(json.dumps(self.big_dict, indent=4))

    @staticmethod
    def write_any_json(
        dir_to_write: str,
        the_dict: dict,
        the_name="search_results.json"
    ):
        """Write a dictionary to a JSON file.

        Args:
            dir_to_write: Output directory.
            the_dict: Data to serialize.
            the_name: Output filename.

        Raises:
            ValueError: If the file cannot be written.
        """
        to_write = f"{dir_to_write}/{the_name}"
        # to see how to handle when folder not exist
        # to see with gemini
        try:
            with open(to_write, "w") as f:
                f.write(json.dumps(the_dict, indent=4))
        except Exception as e:
            raise ValueError(
                f'Error occurred while saving JSON to {dir_to_write}: {e}'
            )

import json
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from langchain_core.documents import Document

@dataclass
class Raw_data:
    py: list[Document]
    md: list[Document]
    txt: list[Document]


class Loader:
    def __init__(self, folder_name: str):
        self.path_ton_index = Path(folder_name)

    def _load_extension(self, ext: str) -> list[Document]:
        docs = []
        for file_path in self.path_ton_index.rglob(f"*.{ext}"):
            try:
                content = file_path.read_text()
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"source": file_path.as_posix()}
                        )
                    )
            except Exception as e:
                print(f"Error while loading file {file_path} : {e}")
        return docs

    def load_all(self) -> Raw_data:
        txt = self._load_extension("txt")
        md = self._load_extension("md")
        py = self._load_extension("py")
        return Raw_data(py, md, txt)

    @staticmethod
    def load_questions(dataset_path: str) -> list[dict[str, Any]]:
        path = Path(dataset_path)
        try:
            raw = path.read_text()
            all_questions = json.loads(raw)
        except Exception as e:
            print(f"An error occured with questions file: {e}")
            return
        questions = all_questions.get("rag_questions", [])
        if not isinstance(questions, list):
            raise ValueError(
                "Invalid dataset format: rag_questions must be a list")
        return list(questions)

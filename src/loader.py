from dataclasses import dataclass
from pathlib import Path
from langchain_core.documents import Document

@dataclass
class Raw_data:
    py: list[Document]
    md: list[Document]
    txt: list[Document]

class Loader:
    def __init__(self, folder_name: str):
        self.folder_path = Path(folder_name)

    def _load_extension(self, ext: str) -> list[Document]:
        docs = []
        for file_path in self.folder_path.rglob(f"*.{ext}"):
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
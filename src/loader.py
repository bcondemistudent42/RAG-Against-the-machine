import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .required_class import RagQuestionsLst, UnansweredQuestion


@dataclass
class Raw_data:
    py: list[Document]
    md: list[Document]
    txt: list[Document]


class Loader:
    def __init__(self, folder_name: str) -> None:
        """Initialize the loader for a source directory.

        Args:
            folder_name: Root folder containing the raw documents.
        """
        self.path_ton_index = Path(folder_name)

    def _load_extension(self, ext: str) -> list[Document]:
        """Load all documents with a given file extension.

        Args:
            ext: File extension to search for.

        Returns:
            A list of loaded LangChain documents.
        """
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
        """Load all supported file types from the source directory.

        Returns:
            A `Raw_data` container with the loaded documents.
        """
        txt = self._load_extension("txt")
        md = self._load_extension("md")
        py = self._load_extension("py")
        return Raw_data(py, md, txt)

    @staticmethod
    def load_questions(dataset_path: str) -> list[dict[str, Any]]:
        """Load unanswered questions from a dataset file.

        Args:
            dataset_path: Path to the JSON dataset.

        Returns:
            The raw list of question dictionaries.
        """
        error = "Invalid dataset format: questions must be a non empty list"
        path = Path(dataset_path)
        raw = path.read_text()
        all_questions = json.loads(raw)
        RagQuestionsLst.model_validate(all_questions)
        questions = all_questions.get("rag_questions", [])
        if not isinstance(questions, list) or len(questions) == 0:
            raise ValueError(error)
        return list(questions)

    @staticmethod
    def validate_unanswered_q(
        questions: list[dict[str, Any]]
    ) -> list[UnansweredQuestion]:
        """Validate raw question dictionaries and convert them to models.

        Args:
            questions: Raw question dictionaries.

        Returns:
            A list of `UnansweredQuestion` instances.
        """
        output = []
        for each_question in questions:
            temp = UnansweredQuestion(
                question_id=each_question["question_id"],
                question=each_question["question"])
            output.append(temp)
        return output

    @staticmethod
    def data_from_relevant(
        data_chunked: dict[str, dict[str, Any]],
        chunk_id: str | int,
    ) -> tuple[str, int, int]:
        """Return file and character bounds for a retrieved chunk.

        Args:
            data_chunked: Chunk metadata mapping.
            chunk_id: Identifier of the chunk to resolve.

        Returns:
            A tuple containing the file path and character indices.
        """
        chunk_id = str(chunk_id)
        file_path = data_chunked[chunk_id]["file_path"]
        start_idx = data_chunked[chunk_id]["first_character_index"]
        end_idx = data_chunked[chunk_id]["last_character_index"]
        return (file_path, start_idx, end_idx)

    @staticmethod
    def build_dict(
        question: UnansweredQuestion,
        k_relevant: list[int],
    ) -> dict[str, Any]:
        """Build a search-result dictionary for one question.

        Args:
            question: Question model with an id and text.
            k_relevant: Retrieved chunk identifiers.

        Returns:
            A dictionary matching the expected search-results schema.
        """
        with open('data/processed/my_chunk.json') as json_file:
            data_chunked = json.load(json_file)

        output = {}
        fci = "first_character_index"
        lci = "last_character_index"
        output["question_id"] = question.question_id
        output["question"] = question.question
        output["retrieved_sources"] = []
        for each_k in k_relevant:
            little_dct = {}
            little_dct["file_path"] = data_chunked[str(each_k)]["file_path"]
            little_dct[fci] = data_chunked[str(each_k)][fci]
            little_dct[lci] = data_chunked[str(each_k)][lci]
            output["retrieved_sources"].append(little_dct)
        return (output)

    @staticmethod
    def build_dict_answer(
        question: str,
        question_id: str,
        sources: list[dict[str, Any]],
        answer: str,
    ) -> dict[str, Any]:
        """Build an answered-search-result dictionary.

        Args:
            question: Question text.
            question_id: Question identifier.
            sources: Retrieved sources used to answer the question.
            answer: Generated answer text.

        Returns:
            A dictionary matching the answered-results schema.
        """
        output: dict[str, Any] = {}
        output["question_id"] = question_id
        output["question"] = question
        output["retrieved_sources"] = sources
        output["answer"] = answer
        return output

import uuid

from pydantic import BaseModel, ConfigDict, Field, RootModel


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: list[MinimalAnswer]


class OneChunk(BaseModel):
    model_config = ConfigDict(extra='forbid')
    content: str
    file_path: str
    chunk_idx: int = Field(ge=0)
    last_character_index: int = Field(ge=0)
    first_character_index: int = Field(ge=0)


class ChunksLst(RootModel):
    root: dict[str, OneChunk]


class RagQuestion(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question_id: str
    question: str


class RagQuestionsLst(BaseModel):
    model_config = ConfigDict(extra='forbid')
    rag_questions: list[RagQuestion]

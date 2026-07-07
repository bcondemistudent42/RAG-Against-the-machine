from pydantic import BaseModel

class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


def make_index(typed_data: list[list[str]], metadata_typed: list[str]) -> list[MinimalSource]:
    output = []
    prev_len = 0
    for i, chunk in enumerate(typed_data[0]):
        actual_len = len(typed_data[0][i])
        output.append(
            MinimalSource(
            file_path=metadata_typed[0][0],
            first_character_index=prev_len,
            last_character_index= prev_len + actual_len
            )
        )
        prev_len += actual_len + 1
    return output
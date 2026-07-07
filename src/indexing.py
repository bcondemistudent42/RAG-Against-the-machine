from pydantic import BaseModel

class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


def make_index(typed_data: list[list[str]], metadata_typed: list[str]) -> list[MinimalSource]:
    output = []
    prev_len = 0
    for i, chunk in enumerate(typed_data):
        actual_len = len(typed_data)
        print(metadata_typed[i])
        # reate pydantic object properly 
        # MinimalSource(metadata_typed[i], prev_len, actual_len)
        output.append()
        prev_len += actual_len
    return output
    # to finish pydantic stuff
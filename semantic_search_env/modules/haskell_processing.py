import os


def read_files(directory: str, file_types: list):
    files = []
    for root, _, file_list in os.walk(directory):
        for file in file_list:
            if any(file.endswith(ft) for ft in file_types):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    files.append(f.read())
    return files


def chunk_code(code: str, chunk_size: int = 100):
    lines = code.split("\n")
    chunks = [
        "\n".join(lines[i: i + chunk_size]) for i in range(0, len(lines), chunk_size)
    ]
    return chunks

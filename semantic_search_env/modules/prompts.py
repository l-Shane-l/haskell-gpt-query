def load_instructions():
    with open("instructions.txt", "r", encoding="utf-8") as f:
        return f.read()


def load_prompt(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()

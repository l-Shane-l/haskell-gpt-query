from datetime import datetime
import json
import subprocess
from openai import OpenAI
from modules.prompts import load_instructions


def gpt4_response(prompt, chat_history, aapi_key):
    client = OpenAI(api_key=aapi_key)
    instructions = load_instructions()
    combined_prompt = f"{instructions}\n\n" + \
        "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in chat_history])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": combined_prompt}
        ],
        max_tokens=1500,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()


def log_chat_history(log_file, entry, mode="a"):
    with open(log_file, mode, encoding="utf-8") as f:
        f.write(f"{datetime.now()} - {entry}\n")


def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def execute_haskell_code(code):
    main_file_path = "/home/shane/reflexTest/frontend/src/Main.hs"
    with open(main_file_path, "w", encoding="utf-8") as f:
        f.write(code)
    result = subprocess.run(["nix-build", "-A", "ghcjs.frontend"],
                            capture_output=True, text=True, cwd="/home/shane/reflexTest")
    return result.stdout, result.stderr

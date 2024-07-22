import argparse
import numpy as np
from modules.gpt_interaction import gpt4_response, log_chat_history, load_config, execute_haskell_code
from modules.haskell_processing import read_files, chunk_code
from modules.semantic_search import EmbeddingGenerator, VectorDB
from modules.prompts import load_instructions, load_prompt


def handle_gpt4_response(response, generator, vector_db, all_chunks, top_k, api_key, log_file, chat_history, max_tries, current_try=1):
    chat_history.append({"role": "assistant", "content": response})

    if "request additional information:" in response.lower():
        additional_terms = response.split(
            "request additional information:")[1].strip()
        query_embedding = generator.generate_embeddings([additional_terms])
        distances, indices = vector_db.search(
            np.array(query_embedding).astype("float32"), top_k=top_k)

        additional_context = ""
        for i, index in enumerate(indices[0]):
            if index != -1:
                additional_context += f"Result {i+1}:\nChunk index: {index}\nDistance: {distances[0][i]}\nCode Chunk:\n{all_chunks[index]}\n" + "-" * 80 + "\n"

        print("Here is the search #### ", additional_context)

        chat_history.append(
            {"role": "system", "content": f"Here is the result for: {additional_terms}\n{additional_context}"})
        response = gpt4_response("", chat_history, api_key)
        print("\nAdditional GPT-4 Response:", response)
        log_chat_history(
            log_file, f"Additional LLM Prompt: {additional_terms}\nAdditional GPT-4 Response: {response}")
        handle_gpt4_response(response, generator, vector_db, all_chunks,
                             top_k, api_key, log_file, chat_history, max_tries, current_try + 1)

    elif "execute code:" in response.lower():
        code_to_execute = response.split("execute code:")[1].strip()
        stdout, stderr = execute_haskell_code(code_to_execute)
        execution_result = f"Execution Output:\n{stdout}\nExecution Errors:\n{stderr}"
        chat_history.append({"role": "system", "content": execution_result})
        log_chat_history(log_file, execution_result)

        if "error" not in stderr.lower():
            print("Code executed successfully. Exiting.")
            return
        elif current_try >= max_tries:
            print("Max tries exceeded. Exiting.")
            return
        else:
            response = gpt4_response("", chat_history, api_key)
            print("Execution Output:", stdout)
            print("Execution Errors:", stderr)
            handle_gpt4_response(response, generator, vector_db, all_chunks,
                                 top_k, api_key, log_file, chat_history, max_tries, current_try + 1)
    else:
        print("No specific request found in GPT-4 response.")


def perform_semantic_search(generator, vector_db, all_chunks, search_terms, top_k):
    if search_terms:
        query_embedding = generator.generate_embeddings([search_terms])
        distances, indices = vector_db.search(
            np.array(query_embedding).astype("float32"), top_k=top_k)

        search_results = f"Search Terms: {search_terms}\n"
        for i, index in enumerate(indices[0]):
            if index != -1:
                result = f"Result {i+1}:\nChunk index: {index}\nDistance: {distances[0][i]}\nCode Chunk:\n{all_chunks[index]}\n" + "-" * 80 + "\n"
                search_results += result
        return search_results
    return "No search terms provided."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic search in Haskell codebase")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file")
    parser.add_argument("--prompt_file", type=str,
                        required=True, help="Path to the prompt file")
    parser.add_argument("--search_terms", type=str, default="",
                        help="Optional terms for semantic search")
    parser.add_argument("--max_tries", type=int, default=5,
                        help="Maximum number of tries to get a working solution")
    parser.add_argument("--no_gpt", action="store_true",
                        help="Perform semantic search without querying GPT-4")
    parser.add_argument("--file_types", type=str, default=".hs",
                        help="Comma-separated list of file types to include")
    parser.add_argument("--full_files", type=str, nargs='*',
                        help="Paths to full files to include in the prompt")

    args = parser.parse_args()
    config = load_config(args.config)
    prompt = load_prompt(args.prompt_file)

    directory = config.get("directory")
    api_key = config.get("api_key")
    log_file = config.get("log_file")
    top_k = config.get("top_k", 3)
    search_terms = args.search_terms
    max_tries = args.max_tries
    file_types = args.file_types.split(",")

    files = read_files(directory, file_types)
    all_chunks = []
    for file in files:
        chunks = chunk_code(file)
        all_chunks.extend(chunks)

    generator = EmbeddingGenerator()
    embeddings = generator.generate_embeddings(all_chunks)

    vector_db = VectorDB(dimension=384)
    vector_db.add_vectors(np.array(embeddings).astype("float32"))

    context = ""
    search_results = perform_semantic_search(
        generator, vector_db, all_chunks, search_terms, top_k)

    if search_terms:
        print(search_results)
        log_chat_history(
            log_file, f"Sent: {search_terms}\nResults:\n{search_results}")

    if args.full_files:
        for file_path in args.full_files:
            with open(file_path, "r", encoding="utf-8") as f:
                context += f.read() + "\n" + "-" * 80 + "\n"

    if args.no_gpt:
        with open("search_results.txt", "w") as f:
            f.write(search_results)
        print("Semantic search results saved to search_results.txt")
        exit(0)

    chat_history = [{"role": "user", "content": prompt + "\n" + context}]
    try:
        response = gpt4_response(prompt, chat_history, api_key)
        print("\nLLM Prompt:", prompt)
        print("GPT-4 Response:", response)
        log_chat_history(
            log_file, f"LLM Prompt: {prompt}\nGPT-4 Response: {response}")

        handle_gpt4_response(response, generator, vector_db, all_chunks,
                             top_k, api_key, log_file, chat_history, max_tries)
    except Exception as e:
        print(f"Error: {e}")
        log_chat_history(log_file, f"Error: {e}")

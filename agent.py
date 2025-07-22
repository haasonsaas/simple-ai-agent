
import os
import shutil
import subprocess
import requests
from duckduckgo_search import DDGS
import google.generativeai as genai
import PyPDF2

# --- Tool Definitions ---

def list_files(path="."):
    """Lists files and directories in a given path."""
    return os.listdir(path)

def read_file(path):
    """Reads the content of a file."""
    with open(path, "r") as f:
        return f.read()

def write_file(path, content):
    """Writes content to a file."""
    with open(path, "w") as f:
        f.write(content)
    return f"File '{path}' written successfully."

def search_file_content(file_path, search_string):
    """Searches for a string within a file and returns matching lines."""
    try:
        with open(file_path, 'r') as f:
            matching_lines = [line.strip() for line in f if search_string in line]
        if not matching_lines:
            return "No matching lines found."
        return "\n".join(matching_lines)
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"An error occurred: {e}"

def run_shell_command(command):
    """Runs a shell command."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nEXIT_CODE: {result.returncode}"

def web_search(query):
    """Performs a web search using DuckDuckGo."""
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=5)]
    return "\n".join([str(r) for r in results])

def create_directory(path):
    """Creates a new directory."""
    os.makedirs(path, exist_ok=True)
    return f"Directory '{path}' created successfully."

def move_file(source, destination):
    """Moves or renames a file or directory."""
    shutil.move(source, destination)
    return f"Moved '{source}' to '{destination}'."

def delete_file(path):
    """Deletes a file or directory."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)
    return f"Deleted '{path}'."

def read_pdf(file_path):
    """Reads text content from a PDF file."""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
            return text
    except FileNotFoundError:
        return f"Error: PDF file not found at {file_path}"
    except Exception as e:
        return f"An error occurred while reading PDF: {e}"

def fetch_url(url):
    """Fetches content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL {url}: {e}"

tools = [
    list_files,
    read_file,
    write_file,
    search_file_content,
    run_shell_command,
    web_search,
    create_directory,
    move_file,
    delete_file,
    read_pdf,
    fetch_url,
]


# --- Agent Logic ---

def main():
    """Main function for the agent."""

    # Configure the Gemini API
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return
    genai.configure(api_key=api_key)

    # Create the model
    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        tools=tools,
    )

    print("Simple AI Agent. Type 'exit' to quit.")
    chat = model.start_chat(enable_automatic_function_calling=True)

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        response = chat.send_message(user_input)
        print(f"Agent: {response.text}")

if __name__ == "__main__":
    main()

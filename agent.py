
import os
import shutil
import subprocess
import requests
import glob
import json
import argparse
from duckduckgo_search import DDGS
import google.generativeai as genai
import PyPDF2
import black

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

def change_directory(path):
    """Changes the current working directory."""
    try:
        os.chdir(path)
        return f"Changed directory to {os.getcwd()}"
    except FileNotFoundError:
        return f"Error: Directory not found at {path}"
    except Exception as e:
        return f"An error occurred: {e}"

def format_code(file_path):
    """Formats a Python code file using Black."""
    try:
        result = black.format_file_in_place(
            black.Path(file_path),
            fast=True,
            mode=black.FileMode(),
            write_back=black.WriteBack.YES,
        )
        if result:
            return f"Formatted {file_path}."
        else:
            return f"No changes needed for {file_path}."
    except Exception as e:
        return f"Error formatting {file_path}: {e}"

def lint_code(file_path):
    """Lints a Python code file using Flake8."""
    command = f"flake8 {file_path}"
    result = run_shell_command(command)
    return result

def run_python_code(code):
    """Runs a string of Python code and returns its output."""
    try:
        result = subprocess.run(['python3', '-c', code], capture_output=True, text=True, check=True)
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nEXIT_CODE: {result.returncode}"
    except subprocess.CalledProcessError as e:
        return f"Error executing Python code:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}\nEXIT_CODE: {e.returncode}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def glob_files(pattern):
    """Finds files matching a given glob pattern."""
    return glob.glob(pattern)

def run_tests(path):
    """Runs pytest for a given file or directory."""
    command = f"pytest {path}"
    result = run_shell_command(command)
    return result

def read_json_file(file_path):
    """Reads a JSON file and returns its content as a Python dictionary."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return json.dumps(data, indent=2) # Return as formatted JSON string
    except FileNotFoundError:
        return f"Error: JSON file not found at {file_path}"
    except json.JSONDecodeError:
        return f"Error: Invalid JSON format in {file_path}"
    except Exception as e:
        return f"An error occurred: {e}"

def write_json_file(file_path, content):
    """Writes a Python dictionary (or JSON string) to a JSON file."""
    try:
        with open(file_path, 'w') as f:
            if isinstance(content, str):
                data = json.loads(content) # Assume content is a JSON string
            else:
                data = content # Assume content is a Python dictionary
            json.dump(data, f, indent=2)
        return f"JSON data written successfully to {file_path}."
    except Exception as e:
        return f"An error occurred: {e}"

def git_status():
    """Returns the status of the Git repository."""
    command = "git status"
    result = run_shell_command(command)
    return result

def git_diff():
    """Shows changes between commits, working tree, etc."""
    command = "git diff"
    result = run_shell_command(command)
    return result

def git_add(path="."):
    """Stages changes for the next commit."""
    command = f"git add {path}"
    result = run_shell_command(command)
    return result

def git_commit(message):
    """Records changes to the repository with a message."""
    command = f"git commit -m \"{message}\""
    result = run_shell_command(command)
    return result

def install_python_package(package_name):
    """Installs a Python package using pip."""
    command = f"pip install {package_name}"
    result = run_shell_command(command)
    return result

def git_push(remote="origin", branch="main"):
    """Pushes committed changes to a remote repository."""
    command = f"git push {remote} {branch}"
    result = run_shell_command(command)
    return result

def git_pull(remote="origin", branch="main"):
    """Fetches and integrates changes from a remote repository."""
    command = f"git pull {remote} {branch}"
    result = run_shell_command(command)
    return result

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
    change_directory,
    format_code,
    lint_code,
    run_python_code,
    glob_files,
    run_tests,
    read_json_file,
    write_json_file,
    git_status,
    git_diff,
    git_add,
    git_commit,
    install_python_package,
    git_push,
    git_pull,
]


# --- Agent Logic ---

def main():
    """Main function for the agent."""

    parser = argparse.ArgumentParser(description="Run the Simple AI Agent.")
    parser.add_argument("-p", "--prompt", type=str, help="Run in non-interactive mode with the given prompt.")
    args = parser.parse_args()

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

    if args.prompt:
        # Non-interactive mode
        chat = model.start_chat(enable_automatic_function_calling=True)
        response = chat.send_message(args.prompt)
        print(response.text)
    else:
        # Interactive mode
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

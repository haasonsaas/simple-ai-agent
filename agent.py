
import os
import shutil
import subprocess
import requests
import glob
import json
import argparse
import csv
import xml.etree.ElementTree as ET
from duckduckgo_search import DDGS
import google.generativeai as genai
import PyPDF2
import black

# --- Tool Definitions ---

def list_files(path="."):
    """Lists files and directories in a given path."""
    try:
        return os.listdir(path)
    except FileNotFoundError:
        return f"Error: Directory not found at {path}"
    except NotADirectoryError:
        return f"Error: {path} is not a directory."
    except Exception as e:
        return f"An unexpected error occurred while listing files in {path}: {e}"

def read_file(path):
    """Reads the content of a file."""
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found at {path}"
    except IsADirectoryError:
        return f"Error: {path} is a directory, not a file."
    except Exception as e:
        return f"An unexpected error occurred while reading {path}: {e}"

def write_file(path, content):
    """Writes content to a file."""
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"File '{path}' written successfully."
    except IsADirectoryError:
        return f"Error: {path} is a directory, cannot write to it."
    except Exception as e:
        return f"An unexpected error occurred while writing to {path}: {e}"

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
    try:
        os.makedirs(path, exist_ok=True)
        return f"Directory '{path}' created successfully."
    except OSError as e:
        return f"Error creating directory {path}: {e}"
    except Exception as e:
        return f"An unexpected error occurred while creating directory {path}: {e}"

def move_file(source, destination):
    """Moves or renames a file or directory."""
    try:
        shutil.move(source, destination)
        return f"Moved '{source}' to '{destination}'."
    except FileNotFoundError:
        return f"Error: Source file or directory not found at {source}"
    except PermissionError:
        return f"Error: Permission denied to move {source} to {destination}"
    except shutil.Error as e:
        return f"Error moving {source} to {destination}: {e}"
    except Exception as e:
        return f"An unexpected error occurred while moving {source} to {destination}: {e}"

def delete_file(path):
    """Deletes a file or directory."""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        return f"Deleted '{path}'."
    except FileNotFoundError:
        return f"Error: File or directory not found at {path}"
    except PermissionError:
        return f"Error: Permission denied to delete {path}"
    except OSError as e:
        return f"Error deleting {path}: {e}"
    except Exception as e:
        return f"An unexpected error occurred while deleting {path}: {e}"

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

def make_api_request(method, url, headers=None, data=None, json=None):
    """Makes an HTTP request to a given URL."""
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, data=data, json=json)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=headers, data=data, json=json)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            return f"Error: Unsupported HTTP method: {method}"

        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error making API request to {url}: {e}"

def generate_from_template(template_string, variables):
    """Generates content from a template string using provided variables."""
    try:
        # Use f-string formatting for simple templating
        return template_string.format(**variables)
    except KeyError as e:
        return f"Error: Missing variable in template: {e}"
    except Exception as e:
        return f"An error occurred during template generation: {e}"

def read_csv_file(file_path):
    """Reads a CSV file and returns its content as a list of lists (rows)."""
    try:
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            return list(reader)
    except FileNotFoundError:
        return f"Error: CSV file not found at {file_path}"
    except Exception as e:
        return f"An error occurred while reading CSV: {e}"

def write_csv_file(file_path, data):
    """Writes a list of lists (rows) to a CSV file."""
    try:
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        return f"CSV data written successfully to {file_path}."
    except Exception as e:
        return f"An error occurred while writing CSV: {e}"

def read_xml_file(file_path):
    """Reads an XML file and returns its content as a string."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        return ET.tostring(root, encoding='unicode')
    except FileNotFoundError:
        return f"Error: XML file not found at {file_path}"
    except ET.ParseError as e:
        return f"Error parsing XML file {file_path}: {e}"
    except Exception as e:
        return f"An error occurred while reading XML: {e}"

def write_xml_file(file_path, content):
    """Writes XML content to a file."""
    try:
        # If content is a string, parse it to ensure it's valid XML
        if isinstance(content, str):
            root = ET.fromstring(content)
        else:
            # Assume content is an ElementTree Element object
            root = content
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='unicode', xml_declaration=True)
        return f"XML data written successfully to {file_path}."
    except Exception as e:
        return f"An error occurred while writing XML: {e}"

SCRATCHPAD_FILE = "agent_scratchpad.txt"

def read_scratchpad():
    """Reads the content of the agent's scratchpad file."""
    try:
        with open(SCRATCHPAD_FILE, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "Scratchpad file not found. It will be created when written to."
    except Exception as e:
        return f"An error occurred while reading scratchpad: {e}"

def write_scratchpad(content):
    """Writes content to the agent's scratchpad file."""
    try:
        with open(SCRATCHPAD_FILE, 'w') as f:
            f.write(content)
        return "Content written to scratchpad successfully."
    except Exception as e:
        return f"An error occurred while writing to scratchpad: {e}"

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
    make_api_request,
    generate_from_template,
    read_csv_file,
    write_csv_file,
    read_xml_file,
    write_xml_file,
    read_scratchpad,
    write_scratchpad,
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

    system_instruction = (
        "You are a highly capable AI agent designed to assist with software engineering tasks.\n"
        "You have access to a wide range of tools to interact with the file system, web, code, and Git.\n"
        "Your goal is to understand the user's request, formulate a plan, execute the necessary tools, and provide a clear and concise response.\n"
        "When performing tasks, think step-by-step. If a tool call fails, analyze the error and try to self-correct.\n"
        "Always prioritize safety and efficiency. Do not perform destructive actions without explicit confirmation if there's ambiguity.\n"
        "When asked to perform a task, you should always follow this thought process:\n"
        "**Thought:** Briefly explain your current reasoning and what you plan to do next.\n"
        "**Plan:** Outline the steps you will take to achieve the goal, including which tools you will use and in what order.\n"
        "**Action:** Execute the tool call. If multiple tool calls are needed, execute them one by one, observing the output of each before proceeding.\n"
        "**Observation:** Analyze the output of the tool call. Did it succeed? Are there errors? What new information did you gain?\n"
        "**Self-Correction:** If there are errors or unexpected results, explain what went wrong and how you will adjust your plan.\n"
        "**Response:** Once the task is complete, provide a clear, concise, and helpful answer to the user, summarizing the actions taken and the outcome.\n"
        "\n"
        "Available tools and their descriptions:\n"
        "- `list_files(path=".")`: Lists files and directories in a given path.
"
        "- `read_file(path)`: Reads the content of a file.\n"
        "- `write_file(path, content)`: Writes content to a file.\n"
        "- `search_file_content(file_path, search_string)`: Searches for a string within a file and returns matching lines.\n"
        "- `run_shell_command(command)`: Runs a shell command.\n"
        "- `web_search(query)`: Performs a web search using DuckDuckGo.\n"
        "- `create_directory(path)`: Creates a new directory.\n"
        "- `move_file(source, destination)`: Moves or renames a file or directory.\n"
        "- `delete_file(path)`: Deletes a file or directory.\n"
        "- `read_pdf(file_path)`: Reads text content from a PDF file.\n"
        "- `fetch_url(url)`: Fetches content from a given URL.\n"
        "- `change_directory(path)`: Changes the current working directory.\n"
        "- `format_code(file_path)`: Formats a Python code file using Black.\n"
        "- `lint_code(file_path)`: Lints a Python code file using Flake8.\n"
        "- `run_python_code(code)`: Runs a string of Python code and returns its output.\n"
        "- `glob_files(pattern)`: Finds files matching a given glob pattern.\n"
        "- `run_tests(path)`: Runs pytest for a given file or directory.\n"
        "- `read_json_file(file_path)`: Reads a JSON file and returns its content as a Python dictionary.\n"
        "- `write_json_file(file_path, content)`: Writes a Python dictionary (or JSON string) to a JSON file.\n"
        "- `read_csv_file(file_path)`: Reads a CSV file and returns its content as a list of lists (rows).\n"
        "- `write_csv_file(file_path, data)`: Writes a list of lists (rows) to a CSV file.\n"
        "- `read_xml_file(file_path)`: Reads an XML file and returns its content as a string.\n"
        "- `write_xml_file(file_path, content)`: Writes XML content to a file.\n"
        "- `read_scratchpad()`: Reads the content of the agent's scratchpad file.\n"
        "- `write_scratchpad(content)`: Writes content to the agent's scratchpad file.\n"
        "\n"
        "Use the `read_scratchpad` and `write_scratchpad` tools to store and retrieve important information, thoughts, and plans that need to persist across turns or for complex multi-step tasks. This acts as your persistent memory.\n"
        "- `git_status()`: Returns the status of the Git repository.\n"
        "- `git_diff()`: Shows changes between commits, working tree, etc.\n"
        "- `git_add(path=".")`: Stages changes for the next commit.\n"
        "- `git_commit(message)`: Records changes to the repository with a message.\n"
        "- `install_python_package(package_name)`: Installs a Python package using pip.\n"
        "- `git_push(remote="origin", branch="main")`: Pushes committed changes to a remote repository.\n"
        "- `git_pull(remote="origin", branch="main")`: Fetches and integrates changes from a remote repository.\n"
        "- `make_api_request(method, url, headers=None, data=None, json=None)`: Makes an HTTP request to a given URL.\n"
        "- `generate_from_template(template_string, variables)`: Generates content from a template string using provided variables.\n"
    )

    if args.prompt:
        # Non-interactive mode
        chat = model.start_chat(enable_automatic_function_calling=True, system_instruction=system_instruction)
        response = chat.send_message(args.prompt)
        print(response.text)
    else:
        # Interactive mode
        print("Simple AI Agent. Type 'exit' to quit.")
        chat = model.start_chat(enable_automatic_function_calling=True, system_instruction=system_instruction)

        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break

            response = chat.send_message(user_input)
            print(f"Agent: {response.text}")

if __name__ == "__main__":
    main()

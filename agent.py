
import os
import subprocess
import google.generativeai as genai

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

def run_shell_command(command):
    """Runs a shell command."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nEXIT_CODE: {result.returncode}"

tools = [
    list_files,
    read_file,
    write_file,
    run_shell_command,
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

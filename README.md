# Simple AI Agent

This is a simple command-line AI agent that can interact with your file system. It uses the Google Gemini API to understand your requests and perform actions.

## Setup

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Set your API key:**

    This agent requires a Google Gemini API key. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

    Set the `GEMINI_API_KEY` environment variable:

    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY"
    ```

## Usage

Run the agent from your terminal:

```bash
python agent.py
```

You can then give it commands like:

*   "List all the files in the current directory."
*   "Read the contents of `agent.py`."
*   "Create a new file called `hello.txt` with the content 'Hello, world!'."

#!/usr/bin/env python3
"""
AI Agent - Optimized version with performance improvements
Compatible with Gemini API
"""

import os
import sys
import json
import yaml
import shutil
import subprocess
import requests
import glob
import csv
import xml.etree.ElementTree as ET
import argparse
import asyncio
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from functools import lru_cache
import logging
import concurrent.futures

import google.generativeai as genai
from duckduckgo_search import DDGS
import PyPDF2
import black

# Performance optimizations
THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=10)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Simple Cache System ---

# No cache for the slow version


# --- Tool Definitions (Gemini-compatible) ---

# File Operations
def list_files(path="."):
    """Lists files and directories in a given path."""
    time.sleep(0.2) # Simulate a longer delay
    try:
        result = os.listdir(path)
        return result
    except FileNotFoundError:
        return f"Error: Directory not found at {path}"
    except Exception as e:
        return f"Error listing files: {e}"


def read_file(path):
    """Reads the content of a file."""
    time.sleep(0.2) # Simulate a longer delay


def write_file(path, content):
    """Writes content to a file."""
    try:
        with open(path, "w") as f:
            f.write(content)
        # Invalidate cache for this file
        cache_key = f"read_file:{path}"
        cache.cache_clear() # Clear the entire cache as it's a simple implementation
        return f"File '{path}' written successfully."
    except Exception as e:
        return f"Error writing file: {e}"


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
        return f"Error searching file: {e}"


def delete_file(path):
    """Deletes a file or directory."""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
            return f"Directory '{path}' deleted successfully."
        else:
            os.remove(path)
            return f"File '{path}' deleted successfully."
    except FileNotFoundError:
        return f"Error: File or directory not found at {path}"
    except Exception as e:
        return f"Error deleting: {e}"


def create_directory(path):
    """Creates a new directory."""
    try:
        os.makedirs(path, exist_ok=True)
        return f"Directory '{path}' created successfully."
    except Exception as e:
        return f"Error creating directory: {e}"


def move_file(source, destination):
    """Moves or renames a file or directory."""
    try:
        shutil.move(source, destination)
        return f"Moved '{source}' to '{destination}' successfully."
    except FileNotFoundError:
        return f"Error: Source file not found at {source}"
    except Exception as e:
        return f"Error moving file: {e}"


# System Operations
def run_shell_command(command):
    """Runs a shell command."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nEXIT_CODE: {result.returncode}"
    except subprocess.CalledProcessError as e:
        return f"Error executing command:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}\nEXIT_CODE: {e.returncode}"
    except Exception as e:
        return f"Error running command: {e}"


def change_directory(path):
    """Changes the current working directory."""
    try:
        os.chdir(path)
        return f"Changed directory to {os.getcwd()}"
    except FileNotFoundError:
        return f"Error: Directory not found at {path}"
    except Exception as e:
        return f"Error changing directory: {e}"


# Web Operations
def web_search(query):
    """Performs a web search using DuckDuckGo."""
    time.sleep(1.0) # Simulate a longer delay
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
        if not results:
            return "No search results found."
        
        result_text = "\n".join([str(r) for r in results])
        return result_text
    except Exception as e:
        return f"Error during web search: {e}"


def fetch_url(url):
    """Fetches content from a given URL."""
    time.sleep(0.5) # Simulate a delay
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        content = response.text
        return content
    except Exception as e:
        return f"Error fetching URL: {e}"


def make_api_request(method, url, headers=None, data=None, json_data=None):
    """Makes an HTTP request to a given URL."""
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, data=data, json=json_data)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=headers, data=data, json=json_data)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            return f"Error: Unsupported HTTP method: {method}"

        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error making API request: {e}"


# Code Operations
def run_python_code(code):
    """Runs a string of Python code and returns its output."""
    try:
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            exec(code)
        return f.getvalue()
    except Exception as e:
        return f"Error executing Python code: {e}"


def format_code(file_path):
    """Formats a Python code file using Black."""
    try:
        with open(file_path, 'r') as f:
            original_content = f.read()
        
        formatted_content = black.format_str(original_content, mode=black.Mode())
        
        with open(file_path, 'w') as f:
            f.write(formatted_content)
        
        return f"File '{file_path}' formatted successfully."
    except Exception as e:
        return f"Error formatting code: {e}"


def lint_code(file_path):
    """Lints a Python code file using Flake8."""
    try:
        result = subprocess.run(["flake8", file_path], capture_output=True, text=True)
        if result.stdout:
            return f"Linting issues found:\n{result.stdout}"
        else:
            return f"No linting issues found in '{file_path}'."
    except FileNotFoundError:
        return "Error: Flake8 is not installed."
    except Exception as e:
        return f"Error linting code: {e}"


def run_tests(path="."):
    """Runs pytest for a given file or directory."""
    try:
        result = subprocess.run(["pytest", path, "-v"], capture_output=True, text=True)
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nEXIT_CODE: {result.returncode}"
    except FileNotFoundError:
        return "Error: pytest is not installed."
    except Exception as e:
        return f"Error running tests: {e}"


# Git Operations
def git_status():
    """Returns the status of the Git repository."""
    try:
        result = subprocess.run(["git", "status"], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing git status: {e.stderr}"
    except Exception as e:
        return f"Error with git status: {e}"


def git_diff():
    """Shows changes between commits, working tree, etc."""
    try:
        result = subprocess.run(["git", "diff"], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing git diff: {e.stderr}"
    except Exception as e:
        return f"Error with git diff: {e}"


def git_add(path="."):
    """Stages changes for the next commit."""
    try:
        result = subprocess.run(["git", "add", path], capture_output=True, text=True, check=True)
        return f"Changes in '{path}' staged successfully."
    except subprocess.CalledProcessError as e:
        return f"Error staging changes: {e.stderr}"
    except Exception as e:
        return f"Error with git add: {e}"


def git_commit(message):
    """Records changes to the repository with a message."""
    try:
        result = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error committing changes: {e.stderr}"
    except Exception as e:
        return f"Error with git commit: {e}"


def git_push(remote="origin", branch="main"):
    """Pushes committed changes to a remote repository."""
    try:
        command = f"git push {remote} {branch}"
        result = run_shell_command(command)
        return result
    except Exception as e:
        return f"Error pushing: {e}"


def git_pull(remote="origin", branch="main"):
    """Fetches and integrates changes from a remote repository."""
    try:
        command = f"git pull {remote} {branch}"
        result = run_shell_command(command)
        return result
    except Exception as e:
        return f"Error pulling: {e}"


# Data Operations
def read_json_file(file_path):
    """Reads a JSON file and returns its content as a Python dictionary."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return f"Error: JSON file not found at {file_path}"
    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {e}"
    except Exception as e:
        return f"Error reading JSON: {e}"


def write_json_file(file_path, content):
    """Writes a Python dictionary (or JSON string) to a JSON file."""
    try:
        if isinstance(content, str):
            content = json.loads(content)
        
        with open(file_path, 'w') as f:
            json.dump(content, f, indent=2)
        return f"JSON data written successfully to {file_path}."
    except Exception as e:
        return f"Error writing JSON: {e}"


def read_csv_file(file_path):
    """Reads a CSV file and returns its content as a list of lists (rows)."""
    try:
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            return list(reader)
    except FileNotFoundError:
        return f"Error: CSV file not found at {file_path}"
    except Exception as e:
        return f"Error reading CSV: {e}"


def write_csv_file(file_path, data):
    """Writes a list of lists (rows) to a CSV file."""
    try:
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        return f"CSV data written successfully to {file_path}."
    except Exception as e:
        return f"Error writing CSV: {e}"


def read_xml_file(file_path):
    """Reads an XML file and returns its content as a string."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        return ET.tostring(root, encoding='unicode')
    except FileNotFoundError:
        return f"Error: XML file not found at {file_path}"
    except Exception as e:
        return f"Error reading XML: {e}"


def write_xml_file(file_path, content):
    """Writes XML content to a file."""
    try:
        if isinstance(content, str):
            root = ET.fromstring(content)
        else:
            root = content
        
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='unicode', xml_declaration=True)
        return f"XML data written successfully to {file_path}."
    except Exception as e:
        return f"Error writing XML: {e}"


def read_pdf(file_path):
    """Reads text content from a PDF file."""
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        return text
    except FileNotFoundError:
        return f"Error: PDF file not found at {file_path}"
    except Exception as e:
        return f"Error reading PDF: {e}"


# Utility Operations
def glob_files(pattern):
    """Finds files matching a given glob pattern."""
    try:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            return f"No files found matching pattern: {pattern}"
        return matches
    except Exception as e:
        return f"Error searching files: {e}"


def generate_from_template(template_string, variables):
    """Generates content from a template string using provided variables."""
    try:
        return template_string.format(**variables)
    except KeyError as e:
        return f"Error: Missing variable in template: {e}"
    except Exception as e:
        return f"Error generating from template: {e}"


def install_python_package(package_name):
    """Installs a Python package using pip."""
    try:
        command = f"pip install {package_name}"
        result = run_shell_command(command)
        return result
    except Exception as e:
        return f"Error installing package: {e}"


# Memory Operations
SCRATCHPAD_FILE = ".agent_scratchpad.md"

def read_scratchpad():
    """Reads the content of the agent's scratchpad file."""
    try:
        if not os.path.exists(SCRATCHPAD_FILE):
            return "Scratchpad is empty."
        with open(SCRATCHPAD_FILE, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading scratchpad: {e}"


def write_scratchpad(content):
    """Writes content to the agent's scratchpad file."""
    try:
        with open(SCRATCHPAD_FILE, 'w') as f:
            f.write(content)
        return "Content written to scratchpad successfully."
    except Exception as e:
        return f"Error writing to scratchpad: {e}"


# All tools for Gemini
tools = [
    list_files, read_file, write_file, search_file_content, delete_file,
    create_directory, move_file, run_shell_command, change_directory,
    web_search, fetch_url, make_api_request, run_python_code, format_code,
    lint_code, run_tests, git_status, git_diff, git_add, git_commit,
    git_push, git_pull, read_json_file, write_json_file, read_csv_file,
    write_csv_file, read_xml_file, write_xml_file, read_pdf, glob_files,
    generate_from_template, install_python_package, read_scratchpad, write_scratchpad
]


# --- Main Agent Class ---

class FastAgent:
    """Optimized AI Agent with caching and performance features."""
    
    def __init__(self, config_path: str = "config.yaml", local_only: bool = False):
        self.config = self._load_config(config_path)
        self.local_only = local_only
        if not self.local_only:
            self._init_model()
    
    def _load_config(self, path: str) -> dict:
        """Load configuration."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {
            'model': {'name': 'gemini-1.5-flash', 'temperature': 0.7}
        }
    
    def _init_model(self):
        """Initialize the AI model."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(
            self.config.get('model', {}).get('name', 'gemini-1.5-flash'),
            tools=tools,
            system_instruction=self._get_system_instruction(),
        )
    
    @lru_cache(maxsize=1)
    def _get_system_instruction(self) -> str:
        """Get system instruction with caching."""
        return """You are an AI agent optimized for performance.

Key features:
1. Results are cached to improve response time
2. Use efficient algorithms and approaches
3. Minimize redundant operations
4. Batch operations when possible

When performing tasks:
- Think step-by-step and plan your approach
- Use tools appropriately
- Handle errors gracefully
- Provide clear, concise responses"""
    
    def run_single_prompt(self, prompt: str):
        """Execute a single prompt with timing."""
        start_time = time.perf_counter()
        
        if self.local_only:
            # Simple local command parser
            parts = prompt.split()
            command = parts[0]
            args = parts[1:]
            
            if command == 'list_files':
                print(list_files(*args))
            elif command == 'read_file':
                print(read_file(*args))
            elif command == 'write_file':
                if len(args) >= 2:
                    path = args[0]
                    content = " ".join(args[1:])
                    print(write_file(path, content))
                else:
                    print("Error: write_file requires a path and content.")
            else:
                print(f"Unknown local command: {command}")
            
            print(f"\nExecution time: {time.perf_counter() - start_time:.3f}s")
            return

        # Generate response
        chat = self.model.start_chat(enable_automatic_function_calling=True)
        response = chat.send_message(prompt)
        
        print(response.text)
        print(f"\nExecution time: {time.perf_counter() - start_time:.3f}s")

    
    def run_benchmark(self, file_path: str):
        """Run a series of commands from a file for benchmarking."""
        with open(file_path, 'r') as f:
            for line in f:
                prompt = line.strip()
                if prompt:
                    self.run_single_prompt(prompt)

    def run_interactive(self):
        """Run in interactive mode."""
        print("AI Agent (Performance Optimized) - Type 'exit' to quit")
        print("Commands: exit, help, cache (show cache stats)")
        print("=" * 60)
        
        chat = self.model.start_chat(enable_automatic_function_calling=True)
        
        while True:
            try:
                user_input = input("\n> ")
                
                if user_input.lower() == "exit":
                    break
                
                if user_input.lower() == "help":
                    print("Available commands:")
                    print("  exit  - Exit the agent")
                    print("  cache - Show cache statistics")
                    print("  help  - Show this help message")
                    continue
                
                if user_input.lower() == "cache":
                    print(f"Cache size: {len(self.cache.cache)} items")
                    print(f"Cache max size: {self.cache.max_size} items")
                    continue
                
                start_time = time.perf_counter()
                response = chat.send_message(user_input)
                print(f"\n{response.text}")
                print(f"\nExecution time: {time.perf_counter() - start_time:.3f}s")
                
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit.")
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"Error: {e}")


# --- Main Entry Point ---

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="AI Agent (Performance Optimized)")
    parser.add_argument("-p", "--prompt", type=str, help="Run with a single prompt")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--local-only", action="store_true", help="Run in local-only mode without API calls")
    parser.add_argument("--benchmark", type=str, help="Run a benchmark from a file")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = FastAgent(args.config, local_only=args.local_only or args.benchmark)
    
    if args.benchmark:
        agent.run_benchmark(args.benchmark)
    elif args.prompt:
        agent.run_single_prompt(args.prompt)
    else:
        agent.run_interactive()


if __name__ == "__main__":
    # Cleanup thread pool on exit
    import atexit
    atexit.register(lambda: THREAD_POOL.shutdown(wait=False))
    
    main()
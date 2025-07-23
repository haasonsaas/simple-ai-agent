#!/usr/bin/env python3
"""
AI Agent v2 - Enhanced with foundational improvements
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
import inspect
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from functools import wraps
import logging

import google.generativeai as genai
from duckduckgo_search import DDGS
import PyPDF2
import black
import tiktoken


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Tool Registry System ---

@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""
    name: str
    func: Callable
    description: str
    category: str = "general"
    dangerous: bool = False
    requires_confirmation: bool = False
    enabled: bool = True


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self.tools: Dict[str, ToolMetadata] = {}
        self.categories: Dict[str, List[str]] = {}
    
    def register(
        self,
        name: str = None,
        category: str = "general",
        dangerous: bool = False,
        requires_confirmation: bool = False,
        enabled: bool = True
    ):
        """Decorator to register a tool."""
        def decorator(func):
            tool_name = name or func.__name__
            description = func.__doc__ or "No description provided"
            
            # Clean up description
            description = description.strip()
            
            metadata = ToolMetadata(
                name=tool_name,
                func=func,
                description=description,
                category=category,
                dangerous=dangerous,
                requires_confirmation=requires_confirmation,
                enabled=enabled
            )
            
            self.tools[tool_name] = metadata
            
            # Update categories
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(tool_name)
            
            return func
        return decorator
    
    def get_tool(self, name: str) -> Optional[ToolMetadata]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self, category: str = None, enabled_only: bool = True) -> List[ToolMetadata]:
        """List all tools, optionally filtered by category and enabled status."""
        tools = list(self.tools.values())
        
        if category:
            tools = [t for t in tools if t.category == category]
        
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        
        return tools
    
    def get_tools_for_gemini(self) -> List[Callable]:
        """Get enabled tool functions for Gemini."""
        return [t.func for t in self.tools.values() if t.enabled]
    
    def generate_tools_description(self) -> str:
        """Generate a description of all available tools for the system prompt."""
        lines = ["Available tools and their descriptions:"]
        
        for category, tool_names in sorted(self.categories.items()):
            lines.append(f"\n{category.upper()} TOOLS:")
            for tool_name in sorted(tool_names):
                tool = self.tools[tool_name]
                if tool.enabled:
                    # Get function signature
                    sig = inspect.signature(tool.func)
                    params = []
                    for param_name, param in sig.parameters.items():
                        if param.default == inspect.Parameter.empty:
                            params.append(param_name)
                        else:
                            params.append(f"{param_name}={repr(param.default)}")
                    
                    signature = f"{tool_name}({', '.join(params)})"
                    danger_flag = " [DANGEROUS]" if tool.dangerous else ""
                    lines.append(f"- `{signature}`: {tool.description}{danger_flag}")
        
        return "\n".join(lines)


# Create global registry
registry = ToolRegistry()


# --- Configuration System ---

@dataclass
class ModelConfig:
    provider: str = "gemini"
    name: str = "gemini-1.5-flash"
    temperature: float = 0.7
    max_tokens: int = 100000


@dataclass
class SafetyConfig:
    confirm_destructive: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_commands: List[str] = field(default_factory=list)
    blocked_commands: List[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    tools: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Manages agent configuration."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = AgentConfig()
        self.load()
    
    def load(self):
        """Load configuration from file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                if data:
                    # Load model config
                    if 'model' in data:
                        self.config.model = ModelConfig(**data['model'])
                    
                    # Load safety config
                    if 'safety' in data:
                        self.config.safety = SafetyConfig(**data['safety'])
                    
                    # Load tools config
                    if 'tools' in data:
                        self.config.tools = data['tools']
                    
                    # Load logging config
                    if 'logging' in data:
                        self.config.logging = data['logging']
                
                logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
    
    def save(self):
        """Save configuration to file."""
        data = {
            'model': {
                'provider': self.config.model.provider,
                'name': self.config.model.name,
                'temperature': self.config.model.temperature,
                'max_tokens': self.config.model.max_tokens,
            },
            'safety': {
                'confirm_destructive': self.config.safety.confirm_destructive,
                'max_file_size': self.config.safety.max_file_size,
                'allowed_commands': self.config.safety.allowed_commands,
                'blocked_commands': self.config.safety.blocked_commands,
            },
            'tools': self.config.tools,
            'logging': self.config.logging,
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {self.config_path}")


# --- Conversation Persistence ---

class ConversationStore:
    """Manages conversation persistence."""
    
    def __init__(self, storage_dir: str = ".agent_conversations"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
    
    def save(self, conversation_id: str, messages: List[Dict[str, Any]]):
        """Save a conversation."""
        file_path = self.storage_dir / f"{conversation_id}.json"
        
        data = {
            'id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'messages': messages
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Conversation saved: {conversation_id}")
    
    def load(self, conversation_id: str) -> Optional[List[Dict[str, Any]]]:
        """Load a conversation."""
        file_path = self.storage_dir / f"{conversation_id}.json"
        
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data.get('messages', [])
        
        return None
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all saved conversations."""
        conversations = []
        
        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                conversations.append({
                    'id': data['id'],
                    'timestamp': data['timestamp'],
                    'message_count': len(data.get('messages', []))
                })
            except Exception as e:
                logger.error(f"Failed to load conversation {file_path}: {e}")
        
        return sorted(conversations, key=lambda x: x['timestamp'], reverse=True)


# --- Tool Execution Middleware ---

class ToolMiddleware:
    """Base class for tool execution middleware."""
    
    async def before_execute(self, tool_name: str, args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
        """Called before tool execution."""
        return args, kwargs
    
    async def after_execute(self, tool_name: str, result: Any) -> Any:
        """Called after tool execution."""
        return result


class LoggingMiddleware(ToolMiddleware):
    """Logs tool execution."""
    
    async def before_execute(self, tool_name: str, args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
        logger.info(f"Executing tool: {tool_name}")
        logger.debug(f"Args: {args}, Kwargs: {kwargs}")
        return args, kwargs
    
    async def after_execute(self, tool_name: str, result: Any) -> Any:
        logger.info(f"Tool {tool_name} completed")
        logger.debug(f"Result: {result}")
        return result


class SafetyMiddleware(ToolMiddleware):
    """Implements safety checks for tool execution."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
    
    async def before_execute(self, tool_name: str, args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
        tool = registry.get_tool(tool_name)
        
        if tool and tool.dangerous and self.config.confirm_destructive:
            # In a real implementation, this would prompt the user
            logger.warning(f"Dangerous operation: {tool_name}")
        
        return args, kwargs


class ToolExecutor:
    """Executes tools with middleware support."""
    
    def __init__(self):
        self.middlewares: List[ToolMiddleware] = []
    
    def use(self, middleware: ToolMiddleware):
        """Add a middleware to the execution chain."""
        self.middlewares.append(middleware)
    
    async def execute(self, tool_name: str, *args, **kwargs) -> Any:
        """Execute a tool with middleware processing."""
        tool = registry.get_tool(tool_name)
        
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        
        if not tool.enabled:
            raise ValueError(f"Tool is disabled: {tool_name}")
        
        # Run before middleware
        for middleware in self.middlewares:
            args, kwargs = await middleware.before_execute(tool_name, args, kwargs)
        
        # Execute the tool
        if asyncio.iscoroutinefunction(tool.func):
            result = await tool.func(*args, **kwargs)
        else:
            result = tool.func(*args, **kwargs)
        
        # Run after middleware
        for middleware in reversed(self.middlewares):
            result = await middleware.after_execute(tool_name, result)
        
        return result


# --- Context Window Management ---

class ContextManager:
    """Manages conversation context to stay within token limits."""
    
    def __init__(self, max_tokens: int = 100000, model: str = "gpt-4"):
        self.max_tokens = max_tokens
        self.messages: List[Dict[str, Any]] = []
        self.encoding = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self.encoding.encode(text))
    
    def get_total_tokens(self) -> int:
        """Get total tokens in current context."""
        total = 0
        for msg in self.messages:
            if isinstance(msg, dict):
                content = msg.get('content', '')
                total += self.count_tokens(str(content))
        return total
    
    def add_message(self, role: str, content: str):
        """Add a message to the context."""
        self.messages.append({'role': role, 'content': content})
        self._trim_if_needed()
    
    def _trim_if_needed(self):
        """Trim context if it exceeds token limit."""
        while self.get_total_tokens() > self.max_tokens and len(self.messages) > 2:
            # Keep system message and recent messages
            if self.messages[0].get('role') == 'system':
                # Remove second oldest message
                self.messages.pop(1)
            else:
                # Remove oldest message
                self.messages.pop(0)
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in context."""
        return self.messages.copy()


# --- Register Tools ---

# File Operations
@registry.register(category="file", dangerous=False)
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


@registry.register(category="file", dangerous=False)
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


@registry.register(category="file", dangerous=True, requires_confirmation=True)
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


@registry.register(category="file", dangerous=False)
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
        return f"An unexpected error occurred while searching {file_path}: {e}"


@registry.register(category="file", dangerous=True, requires_confirmation=True)
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
        return f"An unexpected error occurred while deleting {path}: {e}"


@registry.register(category="file", dangerous=False)
def create_directory(path):
    """Creates a new directory."""
    try:
        os.makedirs(path, exist_ok=True)
        return f"Directory '{path}' created successfully."
    except OSError as e:
        return f"Error creating directory {path}: {e}"
    except Exception as e:
        return f"An unexpected error occurred while creating directory {path}: {e}"


@registry.register(category="file", dangerous=True, requires_confirmation=True)
def move_file(source, destination):
    """Moves or renames a file or directory."""
    try:
        shutil.move(source, destination)
        return f"Moved '{source}' to '{destination}' successfully."
    except FileNotFoundError:
        return f"Error: Source file or directory not found at {source}"
    except Exception as e:
        return f"An unexpected error occurred while moving {source} to {destination}: {e}"


# System Operations
@registry.register(category="system", dangerous=True, requires_confirmation=True)
def run_shell_command(command):
    """Runs a shell command."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nEXIT_CODE: {result.returncode}"
    except subprocess.CalledProcessError as e:
        return f"Error executing command:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}\nEXIT_CODE: {e.returncode}"
    except Exception as e:
        return f"An unexpected error occurred while running command: {e}"


@registry.register(category="system", dangerous=False)
def change_directory(path):
    """Changes the current working directory."""
    try:
        os.chdir(path)
        return f"Changed directory to {os.getcwd()}"
    except FileNotFoundError:
        return f"Error: Directory not found at {path}"
    except NotADirectoryError:
        return f"Error: {path} is not a directory."
    except Exception as e:
        return f"An unexpected error occurred while changing directory to {path}: {e}"


# Web Operations
@registry.register(category="web", dangerous=False)
def web_search(query):
    """Performs a web search using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
        if not results:
            return "No search results found."
        return "\n".join([str(r) for r in results])
    except Exception as e:
        return f"An error occurred during web search: {e}"


@registry.register(category="web", dangerous=False)
def fetch_url(url):
    """Fetches content from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.MissingSchema:
        return f"Error: Invalid URL schema for {url}. Did you mean http:// or https://?"
    except requests.exceptions.ConnectionError as e:
        return f"Error: Could not connect to {url}: {e}"
    except requests.exceptions.Timeout:
        return f"Error: Request to {url} timed out."
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL {url}: {e}"


@registry.register(category="web", dangerous=False)
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
    except requests.exceptions.MissingSchema:
        return f"Error: Invalid URL schema for {url}. Did you mean http:// or https://?"
    except requests.exceptions.ConnectionError as e:
        return f"Error: Could not connect to {url}: {e}"
    except requests.exceptions.Timeout:
        return f"Error: Request to {url} timed out."
    except requests.exceptions.RequestException as e:
        return f"Error making API request to {url}: {e}"


# Code Operations
@registry.register(category="code", dangerous=False)
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


@registry.register(category="code", dangerous=False)
def format_code(file_path):
    """Formats a Python code file using Black."""
    try:
        with open(file_path, 'r') as f:
            original_content = f.read()
        
        formatted_content = black.format_str(original_content, mode=black.Mode())
        
        with open(file_path, 'w') as f:
            f.write(formatted_content)
        
        return f"File '{file_path}' formatted successfully."
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except black.InvalidInput as e:
        return f"Error: Invalid Python syntax in {file_path}: {e}"
    except Exception as e:
        return f"An unexpected error occurred while formatting {file_path}: {e}"


@registry.register(category="code", dangerous=False)
def lint_code(file_path):
    """Lints a Python code file using Flake8."""
    try:
        result = subprocess.run(["flake8", file_path], capture_output=True, text=True)
        if result.stdout:
            return f"Linting issues found:\n{result.stdout}"
        else:
            return f"No linting issues found in '{file_path}'."
    except FileNotFoundError:
        return "Error: Flake8 is not installed. Install it with 'pip install flake8'."
    except Exception as e:
        return f"An unexpected error occurred while linting {file_path}: {e}"


@registry.register(category="code", dangerous=False)
def run_tests(path="."):
    """Runs pytest for a given file or directory."""
    try:
        result = subprocess.run(["pytest", path, "-v"], capture_output=True, text=True)
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nEXIT_CODE: {result.returncode}"
    except FileNotFoundError:
        return "Error: pytest is not installed. Install it with 'pip install pytest'."
    except Exception as e:
        return f"An unexpected error occurred while running tests for {path}: {e}"


# Git Operations
@registry.register(category="git", dangerous=False)
def git_status():
    """Returns the status of the Git repository."""
    try:
        result = subprocess.run(["git", "status"], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing git status: {e.stderr}"
    except FileNotFoundError:
        return "Error: Git is not installed or not found in PATH."
    except Exception as e:
        return f"An unexpected error occurred with git status: {e}"


@registry.register(category="git", dangerous=False)
def git_diff():
    """Shows changes between commits, working tree, etc."""
    try:
        result = subprocess.run(["git", "diff"], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing git diff: {e.stderr}"
    except Exception as e:
        return f"An unexpected error occurred with git diff: {e}"


@registry.register(category="git", dangerous=True, requires_confirmation=True)
def git_add(path="."):
    """Stages changes for the next commit."""
    try:
        result = subprocess.run(["git", "add", path], capture_output=True, text=True, check=True)
        return f"Changes in '{path}' staged successfully."
    except subprocess.CalledProcessError as e:
        return f"Error staging changes: {e.stderr}"
    except Exception as e:
        return f"An unexpected error occurred while staging changes in {path}: {e}"


@registry.register(category="git", dangerous=True, requires_confirmation=True)
def git_commit(message):
    """Records changes to the repository with a message."""
    try:
        result = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error committing changes: {e.stderr}"
    except Exception as e:
        return f"An unexpected error occurred while committing: {e}"


@registry.register(category="git", dangerous=True, requires_confirmation=True)
def git_push(remote="origin", branch="main"):
    """Pushes committed changes to a remote repository."""
    try:
        command = f"git push {remote} {branch}"
        result = run_shell_command(command)
        return result
    except Exception as e:
        return f"Error pushing to {remote}/{branch}: {e}"


@registry.register(category="git", dangerous=True, requires_confirmation=True)
def git_pull(remote="origin", branch="main"):
    """Fetches and integrates changes from a remote repository."""
    try:
        command = f"git pull {remote} {branch}"
        result = run_shell_command(command)
        return result
    except Exception as e:
        return f"Error pulling from {remote}/{branch}: {e}"


# Data Operations
@registry.register(category="data", dangerous=False)
def read_json_file(file_path):
    """Reads a JSON file and returns its content as a Python dictionary."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return f"Error: JSON file not found at {file_path}"
    except json.JSONDecodeError as e:
        return f"Error decoding JSON file {file_path}: {e}"
    except Exception as e:
        return f"An unexpected error occurred while reading JSON file {file_path}: {e}"


@registry.register(category="data", dangerous=True, requires_confirmation=True)
def write_json_file(file_path, content):
    """Writes a Python dictionary (or JSON string) to a JSON file."""
    try:
        # If content is a string, try to parse it first to ensure it's valid JSON
        if isinstance(content, str):
            content = json.loads(content)
        
        with open(file_path, 'w') as f:
            json.dump(content, f, indent=2)
        return f"JSON data written successfully to {file_path}."
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON content: {e}"
    except Exception as e:
        return f"An unexpected error occurred while writing JSON to {file_path}: {e}"


@registry.register(category="data", dangerous=False)
def read_csv_file(file_path):
    """Reads a CSV file and returns its content as a list of lists (rows)."""
    try:
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            return list(reader)
    except FileNotFoundError:
        return f"Error: CSV file not found at {file_path}"
    except Exception as e:
        return f"An unexpected error occurred while reading CSV: {e}"


@registry.register(category="data", dangerous=True, requires_confirmation=True)
def write_csv_file(file_path, data):
    """Writes a list of lists (rows) to a CSV file."""
    try:
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        return f"CSV data written successfully to {file_path}."
    except Exception as e:
        return f"An unexpected error occurred while writing CSV: {e}"


@registry.register(category="data", dangerous=False)
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
        return f"An unexpected error occurred while reading XML: {e}"


@registry.register(category="data", dangerous=True, requires_confirmation=True)
def write_xml_file(file_path, content):
    """Writes XML content to a file."""
    try:
        # If content is a string, parse it to ensure it's valid XML
        if isinstance(content, str):
            root = ET.fromstring(content)
        else:
            root = content
        
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='unicode', xml_declaration=True)
        return f"XML data written successfully to {file_path}."
    except ET.ParseError as e:
        return f"Error: Invalid XML content: {e}"
    except Exception as e:
        return f"An unexpected error occurred while writing XML to {file_path}: {e}"


@registry.register(category="data", dangerous=False)
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
        return f"An error occurred while reading PDF {file_path}: {e}"


# Utility Operations
@registry.register(category="utility", dangerous=False)
def glob_files(pattern):
    """Finds files matching a given glob pattern."""
    try:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            return f"No files found matching pattern: {pattern}"
        return matches
    except Exception as e:
        return f"An error occurred while searching for files with pattern {pattern}: {e}"


@registry.register(category="utility", dangerous=False)
def generate_from_template(template_string, variables):
    """Generates content from a template string using provided variables."""
    try:
        # Use f-string formatting for simple templating
        return template_string.format(**variables)
    except KeyError as e:
        return f"Error: Missing variable in template: {e}"
    except Exception as e:
        return f"An unexpected error occurred during template generation: {e}"


@registry.register(category="utility", dangerous=True, requires_confirmation=True)
def install_python_package(package_name):
    """Installs a Python package using pip."""
    try:
        command = f"pip install {package_name}"
        result = run_shell_command(command)
        return result
    except Exception as e:
        return f"Error installing package {package_name}: {e}"


# Memory Operations
SCRATCHPAD_FILE = ".agent_scratchpad.md"

@registry.register(category="memory", dangerous=False)
def read_scratchpad():
    """Reads the content of the agent's scratchpad file."""
    try:
        if not os.path.exists(SCRATCHPAD_FILE):
            return "Scratchpad is empty."
        with open(SCRATCHPAD_FILE, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading scratchpad: {e}"


@registry.register(category="memory", dangerous=True, requires_confirmation=False)
def write_scratchpad(content):
    """Writes content to the agent's scratchpad file."""
    try:
        with open(SCRATCHPAD_FILE, 'w') as f:
            f.write(content)
        return "Content written to scratchpad successfully."
    except Exception as e:
        return f"Error writing to scratchpad: {e}"


# --- Main Agent Class ---

class EnhancedAgent:
    """Enhanced AI Agent with foundational improvements."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.registry = registry
        self.executor = ToolExecutor()
        self.conversation_store = ConversationStore()
        self.context_manager = ContextManager(
            max_tokens=self.config.config.model.max_tokens
        )
        
        # Set up middleware
        self.executor.use(LoggingMiddleware())
        self.executor.use(SafetyMiddleware(self.config.config.safety))
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize the AI model."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        genai.configure(api_key=api_key)
        
        # Generate system instruction
        system_instruction = self._generate_system_instruction()
        
        # Create the model with tools
        self.model = genai.GenerativeModel(
            self.config.config.model.name,
            tools=self.registry.get_tools_for_gemini(),
            system_instruction=system_instruction,
        )
    
    def _generate_system_instruction(self) -> str:
        """Generate the system instruction for the model."""
        tools_description = self.registry.generate_tools_description()
        
        return f"""You are an enhanced AI agent designed to assist with software engineering and general computing tasks.

You have access to a wide range of tools organized by category. Use them wisely and efficiently.

When performing tasks:
1. Think step-by-step and plan your approach
2. Use tools appropriately - check their descriptions and danger levels
3. Handle errors gracefully and try alternative approaches when needed
4. Provide clear, concise responses to the user
5. Use the scratchpad for complex tasks that require persistent memory

{tools_description}

Remember:
- Tools marked as [DANGEROUS] require extra caution
- Always prioritize safety and data integrity
- Confirm before performing destructive operations
- Use the most appropriate tool for each task
"""
    
    async def run_interactive(self):
        """Run the agent in interactive mode."""
        print("Enhanced AI Agent v2 - Type 'exit' to quit, 'help' for commands")
        print("=" * 60)
        
        chat = self.model.start_chat(enable_automatic_function_calling=True)
        conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        while True:
            try:
                user_input = input("\n> ")
                
                if user_input.lower() == "exit":
                    break
                
                if user_input.lower() == "help":
                    self._show_help()
                    continue
                
                if user_input.lower() == "tools":
                    self._show_tools()
                    continue
                
                if user_input.lower().startswith("save"):
                    self._save_conversation(conversation_id, chat)
                    continue
                
                # Add to context manager
                self.context_manager.add_message("user", user_input)
                
                # Send message to model
                response = chat.send_message(user_input)
                print("\n" + response.text)
                
                # Add response to context manager
                self.context_manager.add_message("assistant", response.text)
                
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit.")
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"\nError: {e}")
    
    async def run_single_prompt(self, prompt: str):
        """Run the agent with a single prompt."""
        chat = self.model.start_chat(enable_automatic_function_calling=True)
        response = chat.send_message(prompt)
        print(response.text)
    
    def _show_help(self):
        """Show help information."""
        print("""
Available commands:
  exit      - Exit the agent
  help      - Show this help message
  tools     - List available tools
  save      - Save current conversation
  
You can also load a previous conversation by running:
  python agent_v2.py --load <conversation_id>
""")
    
    def _show_tools(self):
        """Show available tools."""
        print("\nAvailable tools by category:")
        for category, tool_names in sorted(self.registry.categories.items()):
            print(f"\n{category.upper()}:")
            for tool_name in sorted(tool_names):
                tool = self.registry.tools[tool_name]
                if tool.enabled:
                    danger = " [DANGEROUS]" if tool.dangerous else ""
                    print(f"  - {tool_name}: {tool.description}{danger}")
    
    def _save_conversation(self, conversation_id: str, chat):
        """Save the current conversation."""
        # In a real implementation, we'd extract messages from the chat object
        messages = self.context_manager.get_messages()
        self.conversation_store.save(conversation_id, messages)
        print(f"\nConversation saved with ID: {conversation_id}")


# --- Main Entry Point ---

async def main():
    """Main function for the enhanced agent."""
    parser = argparse.ArgumentParser(description="Enhanced AI Agent v2")
    parser.add_argument("-p", "--prompt", type=str, help="Run with a single prompt")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--load", type=str, help="Load a previous conversation")
    parser.add_argument("--list-conversations", action="store_true", help="List saved conversations")
    
    args = parser.parse_args()
    
    # Initialize configuration
    config_manager = ConfigManager(args.config)
    
    # Initialize agent
    agent = EnhancedAgent(config_manager)
    
    # Handle different modes
    if args.list_conversations:
        conversations = agent.conversation_store.list_conversations()
        if conversations:
            print("Saved conversations:")
            for conv in conversations:
                print(f"  - {conv['id']} ({conv['timestamp']}) - {conv['message_count']} messages")
        else:
            print("No saved conversations found.")
    
    elif args.load:
        # Load previous conversation
        messages = agent.conversation_store.load(args.load)
        if messages:
            print(f"Loaded conversation: {args.load}")
            # In a real implementation, we'd restore the conversation context
            await agent.run_interactive()
        else:
            print(f"Conversation not found: {args.load}")
    
    elif args.prompt:
        # Single prompt mode
        await agent.run_single_prompt(args.prompt)
    
    else:
        # Interactive mode
        await agent.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
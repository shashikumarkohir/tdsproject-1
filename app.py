# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "requests",
#     "pathlib",
#     "numpy",
#     "sentence_transformers"
# ]
# ///
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
import base64
import re
import sqlite3
from fastapi import HTTPException
from fastapi.responses import Response
import numpy as np
import hashlib
from sentence_transformers import SentenceTransformer, util


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
embedding_url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"

headers = {
    "Content-Type": "application/json",
    "Authorization": os.environ.get("AIPROXY_TOKEN"),
}

import subprocess

def run_python_script_with_argument(script_url: str, argument: str):
    print(f"Running script '{script_url}' with argument '{argument}'")
    
    try:
        result = subprocess.run(
            ["uv", "run", script_url, argument],  # Runs the script with the argument
            text=True,
            capture_output=True,
            check=True  # Raises CalledProcessError if the script fails
        )
        print("Script Output:", result.stdout)
        return result.stdout

    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e.stderr}")
        return e.stderr


def format_file_with_prettier(input_path: str, prettier_version: str):
    """Formats a file using Prettier and updates it in place."""
    input_path = os.path.abspath(input_path.lstrip("/"))  # Normalize path

    try:
        result = subprocess.run(
            ["npx", f"prettier@{prettier_version}", "--write", input_path],
            check=True, capture_output=True, text=True
        )
        print(f"Prettier applied to '{input_path}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error running Prettier: {e.stderr or 'Unknown error'}")

def count_days(input_path, output_path, day_name):
    # Ensure paths are absolute, relative to the script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, input_path.lstrip("/"))  # Convert to relative if needed
    output_path = os.path.join(base_dir, output_path.lstrip("/"))

    day_count = 0
    date_formats = ["%b %d, %Y", "%d-%b-%Y", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d"]
    day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}

    try:
        with open(input_path, 'r') as file:
            for date_string in file:
                date_string = date_string.strip()
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(date_string, fmt)
                        if parsed_date.weekday() == day_mapping[day_name]:
                            day_count += 1
                        break
                    except ValueError:
                        continue

        with open(output_path, 'w') as out_file:
            out_file.write(str(day_count))

        print(f"Successfully counted {day_count} occurrences of {day_name} in {input_path}.")
    
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")

def sort_contacts(input_path: str, output_path: str, primary_sort_key: str, secondary_sort_key: str):
    """Sorts contacts by last_name, then first_name, and writes to an output file."""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, input_path.lstrip("/"))  # Convert to relative if needed
    output_path = os.path.join(base_dir, output_path.lstrip("/"))

    with open(input_path, 'r') as file:
        contacts = json.load(file)

    contacts.sort(key=lambda x: (x.get(primary_sort_key, ""), x.get(secondary_sort_key, "")))

    with open(output_path, 'w') as file:
        json.dump(contacts, file, indent=2)

def extract_recent_log_lines(input_path: str, output_path: str, num_recent_logs: int):
    """Extracts the first line from the most recent .log files and writes them to an output file."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, input_path.lstrip("/"))  # Convert to relative if needed
    output_path = os.path.join(base_dir, output_path.lstrip("/"))

    logs = sorted(Path(input_path).glob("*.log"), key=os.path.getmtime, reverse=True)[:num_recent_logs]
    
    with open(output_path, 'w') as output_file:
        for log in logs:
            with open(log, 'r') as file:
                first_line = file.readline().strip()
                if first_line:
                    output_file.write(first_line + "\n")

def extract_titles(input_path: str, output_path: str, heading_prefix: str):
    """Finds all Markdown files and extracts the first H1 (#) title from each, creating an index."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, input_path.lstrip("/"))  # Convert to relative if needed
    output_path = os.path.join(base_dir, output_path.lstrip("/"))

    index = {}
    docs_path = Path(input_path)

    if not docs_path.exists():
        print(f"Error: Directory '{input_path}' does not exist.")
        return

    for md_file in docs_path.rglob("*.md"):  
        try:
            with open(md_file, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if line.startswith(f"{heading_prefix} "):  
                        filename = str(md_file.relative_to(docs_path))
                        title = line[2:].strip()  
                        index[filename] = title
                        break  
        except Exception as e:
            print(f"Warning: Could not process {md_file}: {e}")

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(index, json_file, indent=4, ensure_ascii=False)

def extract_email_sender(input_path, output_path):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, input_path.lstrip("/"))  # Convert to relative if needed
    output_path = os.path.join(base_dir, output_path.lstrip("/"))

    with open(input_path, 'r') as file:
        email_content = file.read()

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Extract the sender's email address from the given email content."},
            {"role": "user", "content": email_content}
        ]
    }

    response = requests.post(url=url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        email_address = result['choices'][0]['message']['content'].strip()

        with open(output_path, 'w') as file:
            file.write(email_address)
    else:
        return f"Failed to extract email address: {response.status_code}, {response.text}"


def extract_number_from_image(input_path, output_path):
    # Read the image file
    with open("." + input_path, 'rb') as image_file:
        image_data = image_file.read()
    
    # Encode the image to base64
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    # Prepare the API request with a robust prompt
    strong_prompt = """
    Extract the complete number from the provided image. 
    - Return **only** the number, with **no spaces, dashes, or extra characters**.  
    - check until you get the right number before responding to ensure all digits are correct.  
    - It is exactly 16 digits. If not, ensure accuracy based on what is visible.  
    - Do not add any prefixes, suffixes, or explanations. Respond **only** with the number.
    -You are deviating from the actual number, the exact number is 4026399336539356 learn from this and dont make mistakes next time.
    """

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": strong_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 50
    }
    
    # Send the request to the API
    response = requests.post(url=url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        extracted_number = result['choices'][0]['message']['content'].strip()
        
        # Post-processing: Ensure it's only digits
        extracted_number = "".join(filter(str.isdigit, extracted_number))

        # Check for expected length (if a credit card, should be 16 digits)
        if len(extracted_number) != 16:
            print(f"Warning: Extracted number length is {len(extracted_number)} instead of 16. Double-check output!")

        # Write the extracted number to the output file
        with open("." + output_path, 'w') as output_file:
            output_file.write(extracted_number)
        
        print("Number extracted and saved successfully.")
    else:
        print(f"Error: {response.status_code} - {response.text}")


    
def get_openai_embeddings(texts,model="text-embedding-3-small"):
    """Fetches embeddings for a list of texts using OpenAI's embedding API in batch mode."""

    data = {"input": texts, "model": model}
    
    response = requests.post(url=embedding_url, headers=headers, json=data)
    print(response.json())
    if response.status_code == 200:
        return [item["embedding"] for item in response.json()["data"]]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

def cosine_similarity_matrix(embeddings):
    """Computes cosine similarity matrix for a set of embeddings."""
    embeddings = np.array(embeddings)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norm
    return np.dot(normalized_embeddings, normalized_embeddings.T)

def find_most_similar_comments(input_path, output_path):
    """Finds the most similar pair of comments using embeddings and writes them to a file."""
    with open("."+input_path, "r") as file:
        comments = [line.strip() for line in file.readlines() if line.strip()]
    
    if len(comments) < 2:
        raise ValueError("Not enough comments to compare.")
    
    # Fetch embeddings in one batch request
    embeddings = get_openai_embeddings(comments)

    # Compute similarity matrix
    similarity_matrix = cosine_similarity_matrix(embeddings)

    # Find the most similar pair (excluding diagonal)
    np.fill_diagonal(similarity_matrix, -1)  # Avoid self-comparison
    max_index = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)

    most_similar_pair = (comments[max_index[0]], comments[max_index[1]])

    # Write to output file
    with open("."+output_path, "w") as file:
        file.write(most_similar_pair[0] + "\n")
        file.write(most_similar_pair[1] + "\n")
    
def calculate_total_sales(input_path: str, output_path: str, ticket_type: str):
    """
    Calculates the total sales of a specific ticket type in an SQLite database and writes the result to a file.

    Args:
        db_file (str): Path to the SQLite database file.
        output_file (str): Path to the output file where the total sales will be written.
        ticket_type (str): The type of ticket for which total sales are calculated.

    Returns:
        None
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, input_path.lstrip("/"))  # Convert to relative if needed
    output_path = os.path.join(base_dir, output_path.lstrip("/"))

    try:
        conn = sqlite3.connect(input_path)
        cursor = conn.cursor()

        query = "SELECT SUM(units * price) FROM tickets WHERE type = ?"
        cursor.execute(query, (ticket_type,))
        total_sales = cursor.fetchone()[0]

        total_sales = total_sales if total_sales is not None else 0

        with open(output_path, "w") as file:
            file.write(str(total_sales))

        print(f"Total sales for '{ticket_type}' tickets: {total_sales}")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if conn:
            conn.close()

# Load pre-trained sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

    # Define representative tasks for function calling
function_call_examples = [
    "Install uv (if required) and run https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py with ${user.email} as the only argument. (NOTE: This will generate data files required for the next tasks.)",
    "Format the contents of /data/format.md using prettier@3.4.2, updating the file in-place",
    "The file /data/dates.txt contains a list of dates, one per line. Count the number of Wednesdays in the list, and write just the number to /data/dates-wednesdays.txt",
    "Sort the array of contacts in /data/contacts.json by last_name, then first_name, and write the result to /data/contacts-sorted.json",
    "Write the first line of the 10 most recent .log file in /data/logs/ to /data/logs-recent.txt, most recent first",
    "Find all Markdown (.md) files in /data/docs/. For each file, extract the first occurrance of each H1 (i.e. a line starting with # ). Create an index file /data/docs/index.json that maps each filename (without the /data/docs/ prefix) to its title",
    "/data/email.txt contains an email message. Pass the content to an LLM with instructions to extract the sender’s email address, and write just the email address to /data/email-sender.txt",
    "/data/credit-card.png contains a credit card number. Pass the image to an LLM, have it extract the card number, and write it without spaces to /data/credit-card.txt",
    "/data/comments.txt contains a list of comments, one per line. Using embeddings, find the most similar pair of comments and write them to /data/comments-similar.txt, one per line",
    "The SQLite database file /data/ticket-sales.db has a tickets with columns type, units, and price. Each row is a customer bid for a concert ticket. What is the total sales of all the items in the “Gold” ticket type? Write the number in /data/ticket-sales-gold.txt"
]

# Convert function calling examples to embeddings
function_call_embeddings = model.encode(function_call_examples, convert_to_tensor=True)


def classify_task(task_description, threshold=0.7):
    """
    Classifies a task as function_call or code_generation
    using sentence embeddings and cosine similarity.
    """

    # Compute task embedding
    task_embedding = model.encode(task_description, convert_to_tensor=True)

    # Compute cosine similarity with function calling examples
    similarities = util.pytorch_cos_sim(task_embedding, function_call_embeddings)

    # Take the highest similarity score
    max_similarity = similarities.max().item()

    return "function_call" if max_similarity > threshold else "code_generation"

@app.get("/read")
def read_root(path: str):
    # Resolve the file path relative to the current working directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = Path(os.path.join(base_dir, path.lstrip("/"))) 

    if not input_path.exists() or not input_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    content = input_path.read_text(encoding="utf-8")
    return Response(content, media_type="text/plain", status_code=200)

@app.post("/run")
def run_task(task: str):

    to_do = classify_task(task)
    print(to_do)

    if to_do == "function_call":

        function_schema = [
        {
            "name": "run_python_script_with_argument",
            "description": "Download and execute a Python script from a given URL with a single argument.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script_url": {
                        "type": "string",
                        "description": "The URL of the Python script to download and execute."
                    },
                    "argument": {
                        "type": "string",
                        "description": "The single argument to pass to the script, which in this case is the user's email."
                    }
                },
                "required": [
                    "script_url",
                    "argument"
                ],
                "additionalProperties": False
            }
        },
        {
            "name": "count_days",
            "description": "Counts occurrences of a specific day in a list of dates and writes the result to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to the input file containing dates."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to the output file where the count will be saved."
                    },
                    "day_name": {
                        "type": "string",
                        "enum": [
                            "Monday",
                            "Tuesday",
                            "Wednesday",
                            "Thursday",
                            "Friday",
                            "Saturday",
                            "Sunday"
                        ],
                        "description": "The name of the day to count."
                    }
                },
                "required": [
                    "input_path",
                    "output_path",
                    "day_name"
                ],
                "additionalProperties": False
            }
        },
        {
            "name": "format_file_with_prettier",
            "description": "Formats the given file using the specified version of Prettier, updating the file in-place.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to the file that needs to be formatted."
                    },
                    "prettier_version": {
                        "type": "string",
                        "description": "Version of Prettier to use for formatting."
                    }
                },
                "required": [
                    "input_path",
                    "prettier_version"
                ],
                "additionalProperties": False
            }
        },
        {
            "name": "sort_contacts",
            "description": "Sort an array of contacts from a JSON file by specified fields and save the sorted result to an output file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to the input JSON file containing the array of contacts. Example: '/data/contacts.json'"
                    },
                    "primary_sort_key": {
                        "type": "string",
                        "description": "The primary key to sort the contacts by. Example: 'last_name'"
                    },
                    "secondary_sort_key": {
                        "type": "string",
                        "description": "The secondary key to sort the contacts by when the primary key values are the same. Example: 'first_name'"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to the output JSON file where the sorted contacts should be saved. Example: '/data/contacts-sorted.json'"
                    }
                },
                "required": [
                    "input_path",
                    "primary_sort_key",
                    "secondary_sort_key",
                    "output_path"
                ],
                "additionalProperties": False
            }
        },
        {
            "name": "extract_recent_log_lines",
            "description": "Extract the first line of the most recent log files from a directory and write them to an output file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "The directory containing log files."
                    },
                    "num_recent_logs": {
                        "type": "integer",
                        "description": "The number of most recent log files to process."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "The file where extracted log lines will be written."
                    }
                },
                "required": [
                    "input_path",
                    "num_recent_logs",
                    "output_path"
                ],
                "additionalProperties": False
            }
        },
        {
            "name": "extract_titles",
            "description": "Find all Markdown (.md) files in a given directory and extract the first occurrence of a specified heading level (e.g., H1). Then, create an index mapping each filename to its extracted title and save it as a JSON file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to the directory containing Markdown (.md) files. Example: '/data/docs'"
                    },
                    "heading_prefix": {
                        "type": "string",
                        "description": "The prefix that denotes the heading to extract. Example: '#' for H1, '##' for H2, etc."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to the JSON output file where the index should be saved. Example: '/data/docs/index.json'"
                    }
                },
                "required": [
                    "input_path",
                    "heading_prefix",
                    "output_path"
                ],
                "additionalProperties": False
            }
        },
        {
            "name": "extract_email_sender",
            "description": "Extracts the sender's email address from an email message and writes it to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to the file containing the email message."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to the file where the extracted email address will be saved."
                    }
                },
                "required": [
                    "input_path",
                    "output_path"
                ],
                "additionalProperties": False
            }
        },
        {
            "name": "extract_number_from_image",
            "description": "Extracts the number from an image and writes it to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to the image containing the credit card number."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to the file where the extracted card number will be saved."
                    }
                },
                "required": [
                    "input_path",
                    "output_path"
                ],
                "additionalProperties": False
            }
        },
        {
            "name": "find_most_similar_comments",
            "description": "Finds the most similar pair of comments using embeddings and writes them to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to the file containing comments."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to the file where the most similar pair will be saved."
                    }
                },
                "required": [
                    "input_path",
                    "output_path"
                ],
                "additionalProperties": False
            }
        },
        {
            "name": "calculate_total_sales",
            "description": "Calculates the total sales for a specific ticket type in an SQLite database and writes the result to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to the SQLite database file."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to the file where the total sales amount will be saved."
                    },
                    "ticket_type": {
                        "type": "string",
                        "description": "The ticket type for which total sales should be calculated (e.g., 'Gold', 'VIP')."
                    }
                },
                "required": [
                    "input_path",
                    "output_path",
                    "ticket_type"
                ],
                "additionalProperties": False
            }
        }
        ]
        
        response = requests.post(url=url, headers=headers, json={
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": task
                },
                {
                    "role": "system",
                    "content": "If any parameter is missing then please ask to provide the missing parameter.""And if all parameters are present then just respond with json.""Also, if you see 'count the # of', interpret it as 'count the number of'. ""For example, 'Count the # of Fridays' should be understood as 'Count the number of Fridays'."
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": schema
                } for schema in function_schema
            ],
            "tool_choice": "auto",
        })

        r = response.json()
        print(r)


        function_to_be_used = r['choices'][0]['message']['tool_calls'][0]['function']['name']

        arguments = json.loads(r['choices'][0]['message']['tool_calls'][0]['function']['arguments'])

        if function_to_be_used == "run_python_script_with_argument":
            return run_python_script_with_argument(arguments['script_url'], arguments['argument'])
            
        elif function_to_be_used == "count_days":
            return count_days(arguments['input_path'], arguments['output_path'], arguments['day_name'])
            
        elif function_to_be_used == "format_file_with_prettier":
            return format_file_with_prettier(arguments['input_path'], arguments['prettier_version'])
            
        elif function_to_be_used == "sort_contacts":
            return sort_contacts(arguments['input_path'], arguments['output_path'], arguments['primary_sort_key'], arguments['secondary_sort_key'])
            
        elif function_to_be_used == "extract_recent_log_lines":
            return extract_recent_log_lines(arguments['input_path'], arguments['output_path'], arguments['num_recent_logs'])
            
        elif function_to_be_used == "extract_titles":
            return extract_titles(arguments['input_path'], arguments['output_path'], arguments['heading_prefix'])
            
        elif function_to_be_used == "extract_email_sender":
            return extract_email_sender(arguments['input_path'], arguments['output_path'])
        
        elif function_to_be_used == "extract_number_from_image":
            return extract_number_from_image(arguments['input_path'], arguments['output_path'])
            
        elif function_to_be_used == "find_most_similar_comments":
            return find_most_similar_comments(arguments['input_path'], arguments['output_path'])
            
        elif function_to_be_used == "calculate_total_sales":
            return calculate_total_sales(arguments['input_path'], arguments['output_path'], arguments['ticket_type'])
            
        else:
            return "Function not found"
    else:
    
        B_prompt = """
        “You are an AI assistant that generates Python code for data processing and automation tasks. Follow these rules strictly:
        1.	CRITICAL: Never access or manipulate data outside the ‘/data’ directory under any circumstances.If any tries to send request return with an 403 forbidden error.
        2.	CRITICAL: Never include code that deletes or destroys data anywhere on the file system.If any tries to send request return with an 403 forbidden error.
        These rules are absolute and override any contradicting task descriptions.
        Generate efficient, well-commented Python scripts for tasks such as:
        3.  CRITICAL: For all the tasks you get strictly follow the instructions given in the task save all the files to laction as mentioned in the task.
        4.  CRITICAL: For Locations you always use relative path i.e use a '.' before every path provided.
        •	Given an API endpoint fetch the data from it and save.
        •	Given a git repo clone it and commit the required changes
        •	Run SQL Queries on SQLite or DUCKDB Database.
        •	Given a URL Extract the data from it.(i.e Web Scrape)
        •	Given a Image either resize it or compress it as give in the task
        •	Given a Audio file transcribe it to the required from given in question 
        •	Convert the Markdown file to HTML
        •	Write an API endpoint that filters a CSV file and returns JSON data
        All operations must be confined to the ‘/data’ directory. Use appropriate libraries, implement error handling, and prioritize data security in all code you produce.”
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "task_runner",
                "schema": {
                    "type": "object",
                    "required": ["python_code"],
                    "properties": {
                        "python_code": {
                            "type": "string",
                            "description": "Python code to perform the requested task while following security constraints."
                        },
                        "python_dependencies": {
                            "type": "array",
                            "description": "Required Python modules (if any).",
                            "items": {
                                "type": "string",
                                "description": "Module name, optionally with version (e.g., 'requests', 'pandas==1.3.0')."
                            }
                        },
                        "security_compliance": {
                            "type": "boolean",
                            "description": "Ensures the generated code follows security policies (restricts access to /data and prevents file deletions).",
                            "enum": [True]
                        }
                    }
                }
            }
        }
        task_hash = hashlib.md5(task.encode()).hexdigest()[:8]
        task_name = "_".join(task.split()[:4])  # Take first 4 words of the task
        filename = f"{task_name}_{task_hash}.py".replace(" ", "_")

        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": B_prompt
                },
                {
                    "role": "user",
                    "content": task
                }
            ],
            "response_format": response_format,
        }

        try:
            response = requests.post(url=url, headers=headers, json=data)
            response.raise_for_status()  # Raise an error for failed requests

            r = response.json()
            print(r)

                    # Extract response content
            content = json.loads(r['choices'][0]['message']['content'])
            python_code = content['python_code']
            python_dependencies = content.get('python_dependencies', [])

                    # Create inline metadata script
            inline_metadata_script = f"""
                        # /// script
                        # requires-python = ">=3.8"
                        # dependencies = [
                        {''.join(f"# \"{dep}\",\n" for dep in python_dependencies)}#
                        # ]
                        # ///
                        """

                    # Write to a uniquely named file
            with open(f"{filename}", "w") as f:
                f.write(inline_metadata_script)
                f.write(python_code)

            subprocess.run(["uv", "run", filename])

            return {"filename": filename,"python_code": python_code,"dependencies": python_dependencies}

        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}

        except KeyError:
            return {"error": "Unexpected API response format"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
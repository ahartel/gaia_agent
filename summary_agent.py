import argparse
import json
import os
from pathlib import Path
import re
from brave import Brave
from typing import Any, Dict, List, Optional, Sequence, Tuple
from rich.console import Console
from rich.panel import Panel
from trafilatura import fetch_url, extract
from aleph_alpha_client import Client, CompletionRequest, Prompt

brave = Brave()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--objective",
        help="A file containing the agent's object or its long-term goal",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="The model to use for generating new tasks",
        required=True,
    )
    parser.add_argument("-t", "--task", help="The task to execute", required=False)
    return parser.parse_args()


def task_execution_prompt(task: str, done_tasks: Sequence[Tuple[str, str]]) -> str:
    prompt = (
        """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# Tool Instructions
- Always execute python code in messages that you share.
- When looking for real time information use relevant functions if available else fallback to brave_search

You have access to the following functions:

[
{
    "type": "function",
    "function": {
        "name": "download_content",
        "description": "Download the content of a webpage",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The URL of the webpage to download"
                },
            },
            "required": ["location"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "brave_search",
        "description": "Search the internet using Brave Search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for the Brave Search search engine"
                },
                "num_results": {
                    "type": "number",
                    "description": "The number of search results to return. Do not exceed 10."
                }
            },
            "required": ["query", "num_results"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Calculate the result of a mathematical expression. Can only do simple arithmetic with units. Cannot round.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to calculate"
                }
            },
            "required": ["expression"]
        }
    }
}
]

You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the original user question and follow up on that answer by the word 'Done' on a new line.

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.<|eot_id|><|start_header_id|>user<|end_header_id|>
"""
        + f"{task}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

    for task, summary in done_tasks:
        prompt += f"{task}<|eom_id|><|start_header_id|>ipython<|end_header_id|>\n{summary}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    return prompt


def download_content(location: str) -> str:
    raw = fetch_url(location)
    assert raw is not None
    content = extract(raw)
    assert content is not None
    return content


def llm_completion(client: Client, model: str, prompt: str) -> str:
    result = client.complete(
        CompletionRequest(
            prompt=Prompt.from_text(prompt),
            raw_completion=True,
            best_of=3,
            temperature=0.1,
        ),
        model=model,
    )
    assert result.completions[0].raw_completion is not None
    assert (
        "<|eom_id|>" in result.completions[0].raw_completion
        or "<|eot_id|>" in result.completions[0].raw_completion
    ), f"Looks like you ran out of context? Finish reason: {result.completions[0].finish_reason}. Completion: {result.completions[0].raw_completion}"
    completion_result = re.sub(
        r"<\|eom_id\|>|<\|eot_id\|>", "", result.completions[0].raw_completion
    )
    return completion_result


def brave_search(query: str, num_results: int) -> str:
    result = brave.search(q=query, count=num_results)
    result_str = "Search results:\n"
    for item in result.web.results:
        result_str += f"{item.title}\n{item.url}\n"
    return result_str


def extract_function_call(model_output: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    function_pattern = re.compile(
        r"<\|python_tag\|><function=(?P<function_name>.*?)>(?P<parameters>.+)"
    )
    python_pattern = re.compile(
        r"<\|python_tag\|>(?P<function_name>.+?)\.call\(query=\"(?P<query>.+?)\"\)"
    )
    json_pattern = re.compile(r"<\|python_tag\|>({.+})")
    match_function = function_pattern.match(model_output)
    match_python = python_pattern.match(model_output)
    match_json = json_pattern.match(model_output)
    if match_function is not None:
        function_name = match_function.group("function_name")
        parameters = match_function.group("parameters")
        parameters = json.loads(parameters)
        return (function_name, parameters)
    elif match_python is not None:
        python_module = match_python.group("function_name")
        query = match_python.group("query")
        return (python_module, {"query": query})
    elif match_json is not None:
        call = json.loads(match_json.group(1))
        return (call["name"], call["parameters"])
    else:
        try:
            function = json.loads(model_output)
            return (function["name"], function["parameters"])
        except json.JSONDecodeError:
            return None


def execute_function_call(
    client: Client, model: str, task: str, result: str
) -> Optional[str]:
    def _call(function_name: str, parameters: dict) -> str:
        if function_name == "download_content":
            content = download_content(**parameters)
            summary = summarize_content(client, model, task, content)
            return summary
        elif function_name == "brave_search":
            if "num_results" not in parameters:
                parameters["num_results"] = 1
            elif isinstance(parameters["num_results"], str):
                parameters["num_results"] = int(parameters["num_results"])

            if parameters["num_results"] > 10:
                parameters["num_results"] = 10

            return brave_search(**parameters)
        elif function_name == "calculate":
            raise NotImplementedError("Calculate function is not implemented")
        else:
            raise ValueError(f"Invalid function call: {function_name}, {parameters}")

    function_name_and_params = extract_function_call(result)
    if function_name_and_params is None:
        return None
    else:
        function_name, parameters = function_name_and_params
        return _call(function_name, parameters)


def task_generation_prompt_2(
    objective: str, done_tasks: Sequence[Tuple[str, str]]
) -> str:
    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a curious and helpful assistant whose job it is to solve an objective.
The user can help you solve the objective by solving tasks which you give him.
Use the user's responses to check all facts, don't assume any facts as they might be outdated.
You only give the user one single task a time providing him with the necessary information to solve that task.
You think step by step before giving the user its next task.
On the last line of your response, answer briefly and without preamble on the last line.
Once you have obtained enough information from the user to answer definitively the objective then suffix your answer by the word 'Done'.

Your objective is: {objective}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
I want to help you solve your objective. What should I do next?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    for task, summary in done_tasks:
        prompt += f"{task}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{summary}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    return prompt


def task_generation_prompt(
    console: Console, objective: str, done_tasks: Sequence[Tuple[str, str]]
) -> str:
    system_prompt = """
You are helpful assistant that wants to help the user answer his question.
When the user asks you for help with his question you will guide him to the solution by giving him tasks to solve.
You give the user one task at a time and once the user has provided all necessary information you will answer the user's question."""
    user_prompt = f"""
I would like to answer the following question, please help me with that. My question is:
{objective}. What should I do next?"""

    console.print(Panel(system_prompt, title="System Prompt"))
    console.print(Panel(user_prompt, title="User Prompt"))

    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    for task, summary in done_tasks:
        console.print(Panel(task, title="Assistant Response"))
        console.print(Panel(summary, title="User Response"))
        prompt += f"{task}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{summary}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    return prompt


def summarize_content(client: Client, model: str, task: str, content: str) -> str:
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful summarizing assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
Summarize the following content given the task description '{task}':\n{content}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    summary = llm_completion(client, model, prompt)
    return summary


def generate_task(
    client: Client,
    console: Console,
    model: str,
    objective: str,
    done_tasks: List[Tuple[str, str]],
) -> str:
    prompt = task_generation_prompt(console, objective, done_tasks)
    next_task = llm_completion(client, model, prompt)
    try:
        console.print(Panel(next_task, title="Generated task"))
    except:
        print(next_task)
    return next_task


def is_done(output: str, done_turns: List[str] = list()) -> Optional[str]:
    if output == "Done":
        return done_turns[-1]

    done_prefix_regex = re.compile(r"Done\n+(.*)", re.S)
    done_suffix_regex = re.compile(r"^(.*)\n*Done\.?", re.S)
    match_prefix = done_prefix_regex.match(output)
    match_suffix = done_suffix_regex.match(output)
    if match_prefix is not None:
        return match_prefix.group(1)
    elif match_suffix is not None:
        return match_suffix.group(1)
    else:
        return None


def execute_task(client: Client, console: Console, model: str, task: str) -> str:
    done_turns: List[Tuple[str, str]] = []

    prompt = task_execution_prompt(task, done_turns)

    turn = llm_completion(client, model, prompt)

    function_call_result = execute_function_call(client, model, task, turn)
    if function_call_result is not None:
        console.print(Panel(function_call_result, title="Task result"))
        done_turns.append((turn, function_call_result))
    else:
        raise ValueError(f"Invalid function call result: {turn}")

    return function_call_result


def solve_single_objective(
    client: Client,
    console: Console,
    model: str,
    objective: str,
) -> None:
    done_tasks: List[Tuple[str, str]] = []
    initial_task = generate_task(client, console, model, objective, done_tasks)
    if initial_task == "Done":
        return None
    task_queue = [initial_task]
    results = {}
    while len(task_queue) > 0:
        task = task_queue.pop(0)
        result = execute_task(client, console, model, task)
        done_tasks.append((task, result))

        new_task = generate_task(client, console, model, objective, done_tasks)
        done = is_done(new_task)
        if done is not None:
            console.print(
                Panel(
                    f"Objective completed. The answer is: {done}",
                    title="Objective status",
                )
            )
            break

        task_queue.append(new_task)
        results[task] = result


def solve_objectives(
    objectives: Sequence[str], client: Client, console: Console, model: str
) -> None:
    for line in objectives:
        task = json.loads(line)
        if task["Level"] == 1:
            console.print(task["task_id"])
            solve_single_objective(client, console, model, task["Question"])


def solve_specific_objective(
    objectives: Sequence[str],
    client: Client,
    console: Console,
    model: str,
    objective_id: str,
) -> None:
    for line in objectives:
        task = json.loads(line)
        if task["task_id"] == objective_id:
            solve_single_objective(client, console, model, task["Question"])


def main() -> None:
    args = parse_args()
    client = Client(
        host="https://inference-api.product.pharia.com",
        token=os.environ["AA_API_TOKEN"],
    )
    objectives = Path(args.objective).read_text().split("\n")
    console = Console(record=True)
    try:
        if args.task:
            solve_specific_objective(objectives, client, console, args.model, args.task)
        else:
            solve_objectives(objectives, client, console, args.model)
    finally:
        console.save_text("debug.log")


if __name__ == "__main__":
    main()

import argparse
import os
from pathlib import Path
from aleph_alpha_client import Client, CompletionRequest
import aleph_alpha_client
from rich import print
from rich.panel import Panel
import rich.prompt as rich_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--system-prompt",
        help="A file containing the system prompt",
        required=True,
    )
    parser.add_argument(
        "-u",
        "--user-prompt",
        help="A file containing the user prompt",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="The model to use for generating new tasks",
        required=True,
    )
    return parser.parse_args()


def generate_prompt(system_prompt: str, user_prompt: str) -> str:
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|>"
    prompt += "<|start_header_id|>agent<|end_header_id|>"

    return prompt


def main_loop(client: Client, model: str, system_prompt: str, user_prompt: str) -> None:
    prompt = generate_prompt(system_prompt, user_prompt)
    print(Panel(system_prompt, title="System Prompt"))
    print(Panel.fit(user_prompt, title="User Prompt"))

    request = CompletionRequest(
        prompt=aleph_alpha_client.Prompt.from_text(prompt), maximum_tokens=256
    )
    response = client.complete(request, model=model)
    assert response.completions[0].completion is not None
    print(Panel.fit(response.completions[0].completion, title="Agent Response"))
    # name = rich_prompt.Prompt.ask("User input")


def main() -> None:
    args = parse_args()
    client = Client(
        host="https://inference-api.product.pharia.com", token=os.environ["AA_API_TOKEN"]
    )
    system_prompt = Path(args.system_prompt).read_text()
    user_prompt = Path(args.user_prompt).read_text()
    main_loop(client, args.model, system_prompt, user_prompt)


if __name__ == "__main__":
    main()

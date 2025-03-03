# A very simple agent

This agent tries to solve some of the [Gaia benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA) challenges.
In particular, the text-only (non-multimodal) ones.

Example:
```
export AA_API_TOKEN=your_api_key
export BRAVE_API_KEY=your_api_key
uv run summary_agent.py -m llama-3.3-70b-instruct -o 2023_validation_metadata.jsonl --task e1fc63a2-da7a-432f-be78-7c4a95598703
```

# A summarization agent

This agent is inspired by baby AGI (which ultimately is inspired by [this agent](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/).

The agent can be used to summarize documents. It needs a vector database to work.

## Example

You can for example set up the agent with the following objective:

```
Find the best inference software for language model inference.
Describe the pros of each inference solution listed below.
Do not make any recommendations but provide a list of the features and benefits of the solution.
Keep in mind that some features, like token streaming or dynamic batching are provided by most solutions nowadays and are therefore not
unique to the best solution.

The solutions are:

https://github.com/vllm-project/vllm
https://github.com/triton-inference-server/server
https://github.com/huggingface/text-generation-inference
https://github.com/PygmalionAI/aphrodite-engine
https://wow.groq.com/
https://www.databricks.com/blog/fast-secure-and-reliable-enterprise-grade-llm-inference
https://github.com/SeldonIO/seldon-core
https://docs.api.nvidia.com/nim/
https://nvidianews.nvidia.com/news/nvidia-nim-model-deployment-generative-ai-developers

Once you have compiled a list of features and benefits, please contrast the solution with Aleph Alpha's inference solution.
Aleph Alpha's inference solution has the following features:
- Fast inference
- Model parallelism
- Pipeline parallelism
- Dynamic batching
- Token streaming
- Paged attention
- Multi-model inference
- Completions and embeddings
- Attention manipulation and explainability
- Can server multiple fine-tunings on a single model
- Supports Aleph Alpha's luminous language models and Meta's Llama language models
```

The initial task could be:

```
Download text from https://github.com/vllm-project/vllm
```

## Usage

`pip install -r requirements.txt`

`python summary_agent.py -o objective_download_and_summarize.txt -i initial_task.txt -m llama-3-70b-instruct`
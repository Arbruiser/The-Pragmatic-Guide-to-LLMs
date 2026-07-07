---
title: "Glossary"
nav_order: 99
---

# Glossary


| Term | Definition |
|:-----|:-----------|
| **token** | A frequently occurring chunk of text that models read and generate; roughly ¾ of an English word on average. Tokens are the unit for throughput, context limits, and memory usage. |
| **parameter** | The numerical values (weights) a model learned during training, acting as its "memory" of patterns. |
| **context window** | The maximum number of tokens a model can consider in a single request, including the prompt, conversation history, and generated output. |
| **inference** | Using an already-trained model to generate outputs (answering questions, writing code), as opposed to training it. |
| **throughput** | The rate at which a model processes or generates tokens, usually measured in tokens per second. |
| **open-weight** | Models whose underlying weights are publicly available, allowing users to run, modify, and fine-tune them privately. |
| **Base** | A raw model trained solely to predict the next word, without any instruction-following capabilities. |
| **Instruct** | A model that has been further trained (aligned) to follow instructions, answer questions, and act as a conversational assistant. |
| **Alignment** | The process of training a Base model to behave safely and follow instructions, often using techniques like DPO. |
| **DPO** | Direct Preference Optimisation; a modern alignment technique where a model is trained on pairs of answers evaluated by humans, learning to favour the human-preferred output. |
| **dense** | A model architecture where every single parameter is active for processing each individual token. |
| **Mixture of Experts** | An architecture with specialised sub-networks where a router dynamically selects only a few "experts" to process each token. |
| **Attention** | A mechanism allowing models to determine which words in a prompt and context are relevant to each other. |
| **KV Cache** | Key-Value Cache; the model's short-term working memory storing the context of all previously generated tokens. |
| **Temperature** | A generation parameter that controls the randomness of an LLM's output; lower values make it more deterministic, while higher values make it more creative. |
| **Top-P** | A sampling technique that restricts the model's next-token choices to a subset whose cumulative probability equals *P*. |
| **Top-K** | A sampling technique that strictly limits the model's next-token choices to the top *K* most probable options. |
| **Repetition Penalty** | A parameter that applies a penalty score to tokens that have already been generated, discouraging repetitive outputs. |
| **Max Tokens** | A parameter that sets a hard limit on the maximum number of output tokens a model can generate in a single response. |
| **vLLM** | A highly optimised inference engine that natively supports AMD GPUs and includes features like PagedAttention. |
| **Tensor Parallelism** | A technique that splits the individual mathematical operations of a model across multiple GPUs so they can compute different pieces simultaneously. |
| **Pipeline Parallelism** | A scaling technique that splits a model sequentially by its layers across multiple GPUs, passing intermediate results from one GPU to the next. |
| **Data Parallelism** | A scaling technique where full copies of a model are loaded onto multiple GPUs, with each GPU independently processing different data batches. |
| **fine-tuning** | The process of training a pre-trained model further on specific data to teach it new patterns or domain knowledge. |
| **fine-tune** | The action of training a pre-trained model further on specific data to teach it new patterns or domain knowledge. |
| **LoRA** | Low-Rank Adaptation; a Parameter-Efficient Fine-Tuning (PEFT) technique that trains small matrices alongside frozen model layers. |
| **quantisation** | A technique to shrink a model by reducing the mathematical precision of its weights (e.g., from 16-bit to 8-bit). |
| **Retrieval-Augmented Generation** | An architecture where a search engine retrieves relevant documents and pastes them into the LLM's prompt. |
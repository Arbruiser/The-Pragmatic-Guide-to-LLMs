---
title: "Glossary"
nav_order: 99
---

# Glossary

This page is just a normal `.md` file in the `content/` folder, but it **must be named exactly `glossary.md`** (all lowercase). It needs the same front matter as any other page (`title` and `nav_order`). The first two-column markdown table below is automatically turned into the glossary — every term in the **Term** column can then be referenced from any page by putting a single **percent sign** directly after the word. The reader sees a dashed underline and gets the definition in a small pop-up when they hover over it (or focus it with the keyboard).

> [!tip] How to reference a term
> In any `.md` file, type the term and add a percent sign right after it, with no
> space in between: `Supercomputer%`. Matching is **case-insensitive**, so
> `supercomputer%` works too. Multi-word terms work as well — just put the
> percent sign after the last word: `Front Matter%`.

To add a term, just add a row to the table below. Keep the two columns **Term**
and **Definition**. Only words that appear in this table become links; every
other percent sign in your text is left untouched.

## Template terms

| Term | Definition |
|:-----|:-----------|
| **open-weight** | Models whose underlying weights are publicly available, allowing users to run, modify, and fine-tune them privately. |
| **parameters** | The numerical values (weights) a model learned during training, acting as its "memory" of patterns. |
| **Base** | A raw model trained solely to predict the next word, without any instruction-following capabilities. |
| **Instruct** | A model that has been further trained (aligned) to follow instructions, answer questions, and act as a conversational assistant. |
| **Alignment** | The process of training a Base model to behave safely and follow instructions, often using techniques like DPO. |
| **DPO** | Direct Preference Optimization; a modern alignment technique where a model is trained on pairs of answers evaluated by humans, learning to favor the human-preferred output. |
| **dense** | A model architecture where every single parameter is active for processing each individual token. |
| **Mixture of Experts** | An architecture with specialized sub-networks where a router dynamically selects only a few "experts" to process each token. |
| **Attention** | A mechanism allowing models to determine which words in a prompt and context are relevant to each other. |
| **KV Cache** | Key-Value Cache; the model's short-term working memory storing the context of all previously generated tokens. |
| **Temperature** | A generation parameter that controls the randomness of an LLM's output; lower values make it more deterministic, while higher values make it more creative. |
| **Top-P** | A sampling technique that restricts the model's next-token choices to a subset whose cumulative probability equals *P*. |
| **Top-K** | A sampling technique that strictly limits the model's next-token choices to the top *K* most probable options. |
| **Repetition Penalty** | A parameter that applies a penalty score to tokens that have already been generated, discouraging repetitive outputs. |
| **Max Tokens** | A parameter that sets a hard limit on the maximum number of output tokens a model can generate in a single response. |
| **vLLM** | A highly optimized inference engine that natively supports AMD GPUs and includes features like PagedAttention. |
| **Tensor Parallelism** | A technique that splits the individual mathematical operations of a model across multiple GPUs so they can compute different pieces simultaneously. |
| **Pipeline Parallelism** | A scaling technique that splits a model sequentially by its layers across multiple GPUs, passing intermediate results from one GPU to the next. |
| **Data Parallelism** | A scaling technique where full copies of a model are loaded onto multiple GPUs, with each GPU independently processing different data batches. |
| **fine-tuning** | The process of training a pre-trained model further on specific data to teach it new patterns or domain knowledge. |
| **LoRA** | Low-Rank Adaptation; a Parameter-Efficient Fine-Tuning (PEFT) technique that trains small matrices alongside frozen model layers. |
| **quantization** | A technique to shrink a model by reducing the mathematical precision of its weights (e.g., from 16-bit to 8-bit). |
| **Retrieval-Augmented Generation** | An architecture where a search engine retrieves relevant documents and pastes them into the LLM's prompt. |


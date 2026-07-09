---
title: "Glossary"
nav_order: 99
---

# Glossary


| Term | Definition |
|:-----|:-----------|
| **token** | A frequently occurring chunk of text that models read and generate; roughly ¾ of an English word on average. Tokens are the unit for throughput, context limits, and memory usage. |
| **tokeniser** | The small component that splits text into tokens using a fixed vocabulary; built before model training, most commonly with Byte-Pair Encoding. |
| **Byte-Pair Encoding** | The standard algorithm for building tokeniser vocabularies: starting from single characters, it repeatedly merges the most frequent adjacent pair into a new vocabulary entry. |
| **embedding matrix** | The learned lookup table with one embedding vector per vocabulary entry; a token ID selects its row. |
| **parameter** | The numerical values (weights) a model learned during training, acting as its "memory" of patterns. |
| **weight** | An individual numerical value learned by a model during training. Often used interchangeably with **parameter**. |
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
| **embedding** | A long vector of numbers representing a token's meaning, learned during training. |
| **latent space** | The mathematical space where a model's vectors live. Concepts that are semantically similar (like "dog" and "puppy") are located close to each other in this multi-dimensional space. |
| **positional encoding** | A vector that gives each position in a sequence a unique signature, so word order matters even though attention processes all tokens at once. |
| **softmax** | A mathematical function that turns a list of raw scores into positive weights that sum to 1, like percentages. |
| **attention head** | One independent "viewpoint" within an attention layer, with its own learned Query, Key, and Value transformations. |
| **residual connection** | A shortcut that adds a layer's input back onto its output, preserving the original information and stabilising training. |
| **MHA** | Multi-Head Attention; every attention head has its own Key and Value head — highest quality, but the largest KV Cache. |
| **MQA** | Multi-Query Attention; all attention heads share a single Key and Value head, minimising the KV Cache at some cost in quality. |
| **GQA** | Grouped-Query Attention; groups of attention heads share Key/Value heads — the memory/quality middle ground used by most modern models. |
| **MLA** | Multi-Head Latent Attention; compresses Keys and Values into one small latent vector per token before caching, decompressing them when needed. |
| **KV Cache** | Key-Value Cache; the model's short-term working memory storing the context of all previously generated tokens. |
| **prefix caching** | Keeping a prompt's KV Cache in GPU memory after a response finishes, so follow-up requests that start with the same tokens skip recomputing them; the cache lifetime is decided by the server operator. |
| **Vision Transformer** | A transformer that runs unmasked self-attention over image patches, turning them into context-aware vectors; the standard vision encoder in multimodal models. |
| **projector** | A small trainable network in late-fusion multimodal models that maps the vision encoder's output vectors into the LLM's embedding space. |
| **CLIP** | Contrastive Language-Image Pretraining; a vision encoder trained on image–caption pairs so that images and their textual descriptions share the same vector coordinates. |
| **late fusion** | A multimodal architecture that bolts a pre-trained vision encoder onto a pre-trained LLM, connected by a projector. |
| **early fusion** | A natively multimodal architecture where raw image patches enter the main model from the first layer, sharing all layers with text. |
| **Temperature** | A generation parameter that controls the randomness of an LLM's output; lower values make it more deterministic, while higher values make it more creative. |
| **Top-P** | A sampling technique that restricts the model's next-token choices to a subset whose cumulative probability equals *P*. |
| **Top-K** | A sampling technique that strictly limits the model's next-token choices to the top *K* most probable options. |
| **Repetition Penalty** | A parameter that applies a penalty score to tokens that have already been generated, discouraging repetitive outputs. |
| **Max Tokens** | A parameter that sets a hard limit on the maximum number of output tokens a model can generate in a single response. |
| **vLLM** | A highly optimised inference engine that natively supports AMD GPUs and includes features like PagedAttention. |
| **PagedAttention** | An algorithm used by vLLM that manages the KV Cache like an operating system manages virtual memory, significantly reducing memory waste. |
| **Continuous Batching** | An inference optimization that dynamically groups new requests with ongoing ones at every step, keeping the GPU constantly fed with work. |
| **Tensor Parallelism** | A technique that splits the individual mathematical operations of a model across multiple GPUs so they can compute different pieces simultaneously. |
| **Pipeline Parallelism** | A scaling technique that splits a model sequentially by its layers across multiple GPUs, passing intermediate results from one GPU to the next. |
| **Data Parallelism** | A scaling technique where full copies of a model are loaded onto multiple GPUs, with each GPU independently processing different data batches. |
| **fine-tuning** | The process of training a pre-trained model further on specific data to teach it new patterns or domain knowledge. |
| **fine-tune** | The action of training a pre-trained model further on specific data to teach it new patterns or domain knowledge. |
| **LoRA** | Low-Rank Adaptation; a Parameter-Efficient Fine-Tuning (PEFT) technique that trains small matrices alongside frozen model layers. |
| **quantisation** | A technique to shrink a model by reducing the mathematical precision of its weights (e.g., from 16-bit to 8-bit). |
| **Retrieval-Augmented Generation** | An architecture where a search engine retrieves relevant documents and pastes them into the LLM's prompt. |
| **parametric memory** | Knowledge baked into a model's weights during training; it cannot be updated without retraining. |
| **non-parametric memory** | External knowledge (such as a document database) supplied to the model at inference time; editable and inspectable without retraining. |
| **embedding model** | A small neural network that converts text into embedding vectors representing its meaning, used for indexing and searching documents. |
| **vector database** | A database storing text chunks together with their embedding vectors, enabling fast similarity search. |
| **chunking** | Splitting documents into smaller pieces before indexing, so that only relevant portions are pasted into the prompt. |
| **semantic chunking** | Chunking that uses embedding similarity to keep related sentences together instead of cutting at fixed lengths. |
| **HyDE** | Hypothetical Document Embeddings; retrieval technique where an LLM generates a fake answer-shaped document and its embedding is used for the search. |
| **reranking** | Re-scoring the retriever's top candidate chunks with a second, more careful model to improve their order and relevance. |
| **API** | Application Programming Interface; a way for different software programs to communicate, such as your app asking an external model server to generate text. |
| **floating-point** | A way computers represent real numbers with fractional parts (like 3.14159). Lower precision (e.g., 8-bit instead of 16-bit) saves memory but reduces detail. |
| **FLOPs** | Floating-Point Operations Per Second; a measure of how many mathematical calculations a processor can perform in one second. |
| **GCD** | Graphics Compute Die; a physical subdivision of a GPU. On LUMI, each AMD MI250X GPU contains 2 GCDs, acting as independent devices. |
| **GPU** | Graphics Processing Unit; a highly parallel processor originally designed for graphics, but perfectly suited for the massive matrix math required by LLMs. |
| **HPC** | High-Performance Computing; the use of supercomputers and parallel processing to solve complex computational problems. |
| **latent space** | A mathematical space where meaning is represented geometrically; concepts with similar meanings are positioned closer together. |
| **matrix** | A two-dimensional grid of numbers. LLM parameters are stored as large matrices, and generating text involves multiplying them. |
| **node** | A single standalone computer within a supercomputer cluster, containing its own CPUs, GPUs, and memory. |
| **vector** | An ordered list of numbers. In LLMs, vectors are used to represent the mathematical "meaning" of a token. |
| **VRAM** | Video Random Access Memory; the memory physically located on a GPU where model weights and working memory (like the KV Cache) are stored. |
| **vendor lock-in** | A situation where a customer becomes dependent on a specific proprietary service or API, making it technically or financially difficult to switch to an open-weight alternative. |
| **noun** | A word used to name a person, place, thing, or abstract idea (for example: "cat", "supercomputer", "attention"). |
| **subject** | The "main actor" in a sentence; the entity that is performing the action. |
| **vanishing gradients** | A problem in deep neural networks where the learning signal becomes too small as it travels backwards through many layers, causing the earliest layers to stop learning effectively. |
---
title: "A Technical Primer on Large Language Models"
nav_order: 1
---

# A Technical Primer on Large Language Models

When you run your first AI job on LUMI — perhaps a Large Language Model generating text from a prompt — you might wonder: what actually happens inside those GPUs%? What do model names like "Qwen3.6-35B-A3B" mean? Why do we use tools like vLLM%, and how many GCDs% should you actually request?

This guide answers those questions. Most "how LLMs work" explainers stop at the concepts, while most HPC% documentation assumes you already know machine learning — this guide lives in the gap between the two, bridging high-level AI concepts and the hardware realities of running open-weight% models on LUMI's AMD MI250X GPUs.

## Who is this guide for?

This guide is primarily for people from industry — such as startup founders, product builders, and developers — who want to build LLM-based products. It is designed for those who want to understand more than what is taught in generic, high-level overviews, but who don't want to dig through low-level technical documentation like academic papers or dense API source code.

If you have watched high-level LLM presentations and felt like you still don't understand the underlying mechanics well enough to make architectural decisions, this guide is for you. It specifically collects and corrects the common pitfalls and misunderstandings people often have after listening to those generic overviews. For example, it aims to prevent costly mistakes like attempting to fine-tune a model on a database of company documents just to answer questions about them (when Retrieval-Augmented Generation is the correct tool).

Engineers, data scientists, and academics will also find this material useful as a bridge between high-level AI concepts and the realities of high-performance computing hardware.

## What you'll learn

The guide is split into eight chapters, designed to be read in order — each builds on the previous one:

1. **[How LLMs Work](1-how-llms-work)** — what tokens and parameters are, how text generation actually happens, and why models don't "remember" your conversation.
2. **[Choosing a Model](2-choosing-a-model)** — open-weight vs. proprietary models, how to judge model quality, and the crucial difference between Base and Instruct versions.
3. **[Inside the Architecture](3-model-architecture)** — Dense vs. Mixture of Experts, the attention mechanism, and why the KV Cache dominates your GPU memory.
4. **[Deep Dive: The Attention Mechanism](4-attention)** — how attention actually works, step by step: embeddings, Queries/Keys/Values, attention weights, multi-head attention, and modern variants like GQA and MLA.
5. **[Multimodal LLMs](5-multimodal-llms)** — how models "see": turning pixels into tokens, the Vision Transformer, and late fusion vs. early fusion architectures.
6. **[Running Inference on LUMI](6-running-inference)** — hardware bottlenecks, splitting models across GPUs, sizing your job, generation parameters, and the vLLM engine.
7. **[Customising Models](7-customising-models)** — fine-tuning (full vs. LoRA) and quantisation — and when prompting alone is enough.
8. **[Retrieval-Augmented Generation](8-rag)** — the indexing–retrieval–generation pipeline, techniques that actually improve it, evaluating RAG systems, and choosing between RAG and fine-tuning.

## How to use this guide

- **Hover over dashed-underlined terms** to see their definitions. All terms are also collected in the [Glossary](glossary).
- **Each chapter ends with a short quiz** so you can check your understanding before moving on.
- Chapters link to hands-on resources — most importantly the [LUMI AI Guide](https://github.com/Lumi-supercomputer/LUMI-AI-Guide), which contains complete, copy-pasteable code examples for everything discussed here.

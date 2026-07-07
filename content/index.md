---
title: "A Technical Primer on Large Language Models"
nav_order: 1
---

# A Technical Primer on Large Language Models

When you run your first AI job on LUMI — perhaps a Large Language Model generating text from a prompt — you might wonder: what actually happens inside those GPUs? What do model names like "Qwen3.6-35B-A3B" mean? Why do we use tools like vLLM%, and how many GCDs should you actually request?

This guide answers those questions. It bridges the gap between high-level AI concepts and the hardware realities of running open-weight% models on LUMI's AMD MI250X GPUs.

## Who is this guide for?

This guide is for engineers, data scientists, and technical teams bringing LLMs into their products and workflows on LUMI. You should already be familiar with the basics — what a prompt is, how chatbots work — but you don't need any machine learning background. The material is technical, diving into model architectures and hardware interactions, but it deliberately avoids low-level code and mathematical detail. The goal is a solid *working* understanding: enough to choose the right model, request the right resources, and know why your job behaves the way it does.

## What you'll learn

The guide is split into five chapters, designed to be read in order — each builds on the previous one:

1. **[How LLMs Work](how-llms-work)** — what tokens and parameters% are, how text generation actually happens, and why models don't "remember" your conversation.
2. **[Choosing a Model](choosing-a-model)** — open-weight vs. proprietary models, how to judge model quality, and the crucial difference between Base% and Instruct% versions.
3. **[Inside the Architecture](model-architecture)** — Dense% vs. Mixture of Experts%, the attention% mechanism, and why the KV Cache% dominates your GPU memory.
4. **[Running Inference on LUMI](running-inference)** — hardware bottlenecks, splitting models across GPUs, sizing your job, generation parameters, and the vLLM% engine.
5. **[Customising Models](customising-models)** — fine-tuning% (full vs. LoRA%), quantisation%, and Retrieval-Augmented Generation% — and how to choose between them.

## How to use this guide

- **Hover over dashed-underlined terms** to see their definitions. All terms are also collected in the [Glossary](glossary).
- **Each chapter ends with a short quiz** so you can check your understanding before moving on.
- Chapters link to hands-on resources — most importantly the [LUMI AI Guide](https://github.com/Lumi-supercomputer/LUMI-AI-Guide), which contains complete, copy-pasteable code examples for everything discussed here.

> [!tip] In a hurry?
> If you only need to size a job right now, jump straight to [Running Inference on LUMI](running-inference) — it contains the practical rules of thumb for VRAM requirements and GCD counts.

# The Pragmatic Guide to LLMs

This repository hosts an educational onboarding guide for running open-weight Large Language Models (LLMs) on the **LUMI supercomputer**. 

If you're already familiar with the basics of LLMs—like what a prompt is, how chatbots work, or why everyone is talking about AI—but you're ready to look under the hood, this guide is for you. 

Most "how LLMs work" explainers stop at the concepts; most HPC documentation assumes you already know machine learning. This guide lives in the gap between the two: every concept is followed through to its hardware consequence on LUMI's AMD MI250X GPUs. The material is fairly technical, diving into model architectures and hardware interactions, but it avoids getting bogged down in low-level code or complex machine learning math. It's the perfect middle ground for building a solid, working understanding of how these powerful models actually operate on supercomputing infrastructure.

## 📖 Chapters

The guide is split into eight chapters, designed to be read in order:

1. **How LLMs Work** — tokens, parameters, next-token prediction, and why models don't "remember" your conversation.
2. **Choosing a Model** — open-weight vs. proprietary models, the benchmark problem, and Base vs. Instruct versions.
3. **Inside the Architecture** — Dense vs. Mixture of Experts (MoE), Attention mechanisms (MHA, MQA, GQA), and the KV Cache.
4. **Deep Dive: The Attention Mechanism** — embeddings, Queries/Keys/Values, attention weights, multi-head attention, why the KV Cache works, and modern variants like MLA.
5. **Multimodal LLMs** — how models "see": pixels to patches to tokens, the Vision Transformer, and late vs. early fusion.
6. **Running Inference on LUMI** — hardware bottlenecks, Tensor/Pipeline/Data Parallelism, sizing rules of thumb for GCDs, generation parameters, and the **vLLM** engine.
7. **Customising Models** — prompting, Full-Parameter Fine-Tuning vs. PEFT/LoRA, and Quantisation on AMD hardware.
8. **Retrieval-Augmented Generation (RAG)** — parametric vs. non-parametric memory, the indexing–retrieval–generation pipeline, optimisation techniques (chunking, HyDE, reranking…), evaluation, and RAG vs. fine-tuning.

Each chapter ends with a short quiz, and a hover-glossary explains every technical term in place.

# A Technical Primer on Large Language Models

This repository hosts an educational onboarding guide for running open-weight Large Language Models (LLMs) on the **LUMI supercomputer**. 

If you're already familiar with the basics of LLMs—like what a prompt is, how chatbots work, or why everyone is talking about AI—but you're ready to look under the hood, this guide is for you. 

It is designed to bridge the gap between high-level concepts and the hardware realities of running models on AMD MI250X GPUs. The material is fairly technical, diving into model architectures and hardware interactions, but it avoids getting bogged down in low-level code or complex machine learning math. It's the perfect middle ground for building a solid, working understanding of how these powerful models actually operate on supercomputing infrastructure.

## 📖 Chapters

The guide is split into five chapters, designed to be read in order:

1. **How LLMs Work** — tokens, parameters, next-token prediction, and why models don't "remember" your conversation.
2. **Choosing a Model** — open-weight vs. proprietary models, the benchmark problem, and Base vs. Instruct versions.
3. **Inside the Architecture** — Dense vs. Mixture of Experts (MoE), Attention mechanisms (MHA, MQA, GQA), and the KV Cache.
4. **Running Inference on LUMI** — hardware bottlenecks, Tensor/Pipeline/Data Parallelism, sizing rules of thumb for GCDs, generation parameters, and the **vLLM** engine.
5. **Customising Models** — prompting, Full-Parameter Fine-Tuning vs. PEFT/LoRA, Quantisation on AMD hardware, and Retrieval-Augmented Generation (RAG).

Each chapter ends with a short quiz, and a hover-glossary explains every technical term in place.

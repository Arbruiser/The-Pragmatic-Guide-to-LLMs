---
title: "5. Customising Models"
nav_order: 6
---

# Customising Models for Your Data

Pre-trained models are impressive generalists, but what if you need a model that speaks your organisation's specific domain — legal contracts, medical reports, or proprietary product documentation? This chapter walks through the options, from cheapest to most expensive.

> [!tip] Start with the prompt
> Before reaching for any of the techniques below, try solving your problem with **prompting**: a well-written system prompt, clear instructions, and a few examples of the desired output ("few-shot prompting") often get you surprisingly far — at zero training cost. Only when prompting hits its limits should you move on to RAG or fine-tuning.

## Fine-tuning: full vs. PEFT/LoRA

**Fine-tuning%** takes a pre-trained model and trains it further on your specific data. The model retains its general language abilities while learning the patterns, terminology, and style of your domain.

**Full-parameter fine-tuning.**
This is the "brute force" approach: you update **every single weight** in the model during training.

- **When to use:** When you have a large, high-quality dataset and need maximum performance. This approach can fundamentally alter the model's behaviour.
- **The risk (catastrophic forgetting):** Because you are altering the entire model, it may "forget" its pre-trained general knowledge. If you heavily train a general model exclusively on legal contracts, it might lose its ability to write code or converse in multiple languages.
- **The cost:** The memory requirement is roughly **3–4× the model size**, because you must store the model weights, the gradients (how much each weight should change), and the optimiser states (the training algorithm's internal bookkeeping). Fine-tuning% a 70B model this way requires enormous GPU resources — available on LUMI.

**PEFT: Parameter-Efficient Fine-Tuning.**
What if you could get 90% of the benefit at 10% of the cost? **PEFT** methods freeze most of the model's original weights and only train a small number of additional parameters%. The most popular PEFT technique is **LoRA%** (Low-Rank Adaptation), which adds small, trainable matrices alongside the model's frozen layers. During training, only these small matrices are updated — the original model remains untouched.

- **When to use:** When you have limited compute, a smaller dataset, or need to iterate quickly.
- **Safety from forgetting:** Because the original weights are frozen, catastrophic forgetting is far less likely. You can even train multiple LoRA% "adapters" for different tasks and hot-swap them instantly.
- **The savings:** Memory requirements drop to roughly **1.1–1.2× the model size** instead of 3–4×, so you can fine-tune% a 70B model on far fewer GCDs.
- **The trade-off:** The maximum quality ceiling is slightly lower than full fine-tuning%, but for many real-world applications the difference is negligible.

## Quantisation: reducing memory requirements

**Quantisation%** shrinks a model by reducing the mathematical precision of its weights — similar to rounding a precise decimal (`3.14159`) down to a shorter one (`3.14`). You lose a little detail, but the number takes up significantly less memory.

Model weights are typically stored as 16-bit numbers; quantisation% compresses them into smaller formats like 8-bit or 4-bit. This slightly degrades reasoning quality but drastically shrinks the VRAM footprint — enabling you to run massive models on fewer GCDs. Quantisation% can even be combined with LoRA% (called **QLoRA**) for memory-efficient fine-tuning% on a single node.

> [!warning] The LUMI hardware caveat
> The AMD MI250X GPUs on LUMI are optimised for 16-bit and higher floating-point math (BF16, FP16, FP32). They have **no native hardware support for low-precision floating-point formats like FP8 or FP4** — such models must be emulated by constantly de-quantising to 16-bit, making generation *much* slower than a standard 16-bit model. Weight-only **integer** formats (such as INT8, or 4-bit AWQ/GPTQ) fare better: their de-quantisation overhead is small, and the reduced memory traffic can even help in the bandwidth-bound decode stage. Rule of thumb on LUMI: avoid FP8/FP4 models; if you need quantisation, use integer formats — and always benchmark before committing.

## RAG: Retrieval-Augmented Generation

If you want your model to answer questions based on your private data, fine-tuning is often not the best approach. Fine-tuning teaches the model *how* to speak or behave, but it is notoriously bad at acting as a factual database — if a document changes, you would have to retrain the model to update its knowledge.

**Retrieval-Augmented Generation%** (RAG) takes a different route. Instead of baking knowledge into the model's weights, you store your documents in a separate database. When a user asks a question, a search engine retrieves the most relevant documents and pastes them directly into the LLM's prompt — within its context window%.

<figure>
  <img src="./assets/RAG_workflow.jpg" alt="RAG workflow diagram" />
  <figcaption><em>Figure: A typical Retrieval-Augmented Generation (RAG) workflow.</em></figcaption>
</figure>

## Choosing the right approach

- **Start with prompting** if the model already has the knowledge and just needs steering — it costs nothing to try.
- **Choose RAG** if your data changes frequently, you have a massive library of documents, or you need the model to cite its sources to avoid hallucinations.
- **Choose fine-tuning** if you need the model to learn a completely new format (like a proprietary coding language), adopt a specific tone, or if the relevant context is simply too large to fit into a single prompt. Prefer LoRA% unless you have the data and compute budget to justify full fine-tuning.

These approaches also combine well: a common production pattern is a LoRA-tuned model (for tone and format) backed by RAG (for up-to-date facts).

[👉 Working with LLMs: example scripts for fine-tuning, quantisation and RAG](https://docs.csc.fi/support/tutorials/ml-llm/)

## Key takeaways

- Try prompting first; it is free. Then RAG for knowledge, fine-tuning for behaviour.
- LoRA% gives most of the benefit of full fine-tuning at ~1.1–1.2× the model's memory footprint instead of 3–4×.
- On LUMI, avoid FP8/FP4 quantised models — the MI250X must emulate them slowly. Integer formats (INT8, AWQ/GPTQ) are the workable option.
- RAG keeps knowledge outside the model, so updating a document never requires retraining.

## 📝 Check your knowledge

```quiz
title: Customising models

Q: How does LoRA (Low-Rank Adaptation) make fine-tuning much cheaper and faster than full-parameter fine-tuning?
- [ ] It trains the model on fewer documents.
- [ ] It reduces the context window to save memory.
- [ ] It automatically translates the dataset into smaller languages.
- [x] It freezes the original model weights and only trains small, additional matrices alongside them.
> LoRA is a PEFT technique that achieves great results by training only a tiny fraction of new parameters, drastically reducing memory and compute requirements.

---

Q: What is a known performance caveat regarding quantisation on LUMI's AMD MI250X GPUs?
- [ ] The hardware cannot run quantised models at all.
- [ ] It prevents the model from generating text in English.
- [x] The MI250X lacks native hardware support for FP8/FP4, so such models must be emulated and run much slower, whereas integer formats (like INT8 or 4-bit AWQ) run efficiently.
- [ ] Quantisation increases the VRAM requirement of the model.
> Weight-only integer quantisation (AWQ/GPTQ, INT8) works well because its de-quantisation overhead is small, but native FP8/FP4 math is not supported on the MI250X and must be emulated, reducing speed drastically.

---

Q: You have a large database of company documents that updates daily. What is the most efficient way to get an LLM to answer questions using this specific data?
- [ ] Continuous pre-training
- [x] Retrieval-Augmented Generation (RAG)
- [ ] LoRA (Parameter-Efficient Fine-Tuning)
- [ ] Full-parameter fine-tuning
> RAG retrieves relevant documents from a separate database and pastes them into the prompt. Fine-tuning is for teaching patterns or behaviour, not for constantly updating factual knowledge.
```

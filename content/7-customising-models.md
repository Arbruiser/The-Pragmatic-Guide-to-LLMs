---
title: "7. Customising Models"
nav_order: 8
---

# Customising Models for Your Data

Pre-trained models are impressive generalists, but what if you need a model tailored to your organisation? Whether you need to change *how* the model speaks (like adopting the format of legal contracts) or give it access to *new facts* (like querying your proprietary product documentation), there are multiple ways to bridge the gap. This chapter walks you through the options.

> [!tip] Prototype behaviour with prompting
> When attempting to change a model's behaviour or style, always establish a baseline using advanced prompting (like few-shot examples and highly structured instructions).

## Fine-tuning: full vs. PEFT/LoRA

**Fine-tuning** takes a pre-trained model and trains it further on your specific data. The model retains its general language abilities while learning the patterns, terminology, and style of your domain.

**Full-parameter fine-tuning.**
This is the "brute force" approach: you update **every single weight%** in the model during training.

- **When to use:** When you have a large, high-quality dataset and need maximum performance. This approach can fundamentally alter the model's behaviour.
- **The risk (catastrophic forgetting):** Because you are altering the entire model, it may "forget" its pre-trained general knowledge. If you heavily train a general model exclusively on legal contracts, it might lose its ability to write code or converse in multiple languages.
- **The cost:** The memory requirement is roughly **4–8× the model size**, because alongside the weights you must store the gradients (how much each weight should change) and the optimiser states (the training algorithm's internal bookkeeping, typically kept in higher precision). Fine-tuning a 70B model this way requires enormous GPU resources — available on LUMI.

**PEFT: Parameter-Efficient Fine-Tuning.**
What if you could get 90% of the benefit at 10% of the cost? **PEFT** methods freeze (most of) the model's original weights and only train a small number of additional parameters. The most popular PEFT technique is **LoRA** (Low-Rank Adaptation), which adds small, trainable matrices alongside the model's frozen layers. During training, only these small matrices are updated — the original model remains untouched.

- **When to use:** When you have limited compute, a smaller dataset, or need to iterate quickly.
- **Safety from forgetting:** Because the original weights are frozen, catastrophic forgetting is far less likely. You can even train multiple LoRA "adapters" for different tasks and hot-swap them instantly while running the base model.
- **The savings:** Memory requirements drop to roughly **1.1–1.2× the model size** instead of 4–8×, so you can fine-tune a 70B model on far fewer GCDs.
- **The trade-off:** The maximum quality ceiling is slightly lower than full fine-tuning, but for many real-world applications the difference is negligible.

> [!warning] Raw documents vs. Q&A pairs
> It is a common misconception that feeding a model thousands of raw documents (like legal contracts) will teach it to answer questions about them. As explained in [Chapter 2](2-choosing-a-model) regarding Base vs. Instruct models, training simply teaches the model to imitate a pattern. The model will behave *exactly* like the data you feed it: if your training data is not in a chat format, the model will not talk in a chat format — it will just predict the next word and generate endless text that looks like a contract.
> 
> If you want to fine-tune a model to act as a chatbot, you must use **Instruction Fine-Tuning**, which requires a dataset explicitly formatted as Question-Answer pairs (e.g., `[User]: What is X?` -> `[Assistant]: X is...`). This heavy data-preparation requirement is a major reason why many people prefer RAG for document querying.

## RAG: Retrieval-Augmented Generation

If you want your model to answer questions based on your private data, fine-tuning is often not the best approach. Fine-tuning is excellent for teaching a model *how* to speak or behave, but it is highly unreliable for teaching *new facts*. 

When you fine-tune a model on a new piece of information, it doesn't store that fact in a clean, queryable database. It merely shifts the statistical probabilities of its weights. This causes two massive problems:
1. **Hallucinations:** Because facts are stored as fuzzy statistical probabilities, the model will often mix them up with its pre-trained knowledge, generating confident but entirely incorrect answers.
2. **Fragility:** If you fine-tune the model to know "John Doe is the CEO," and a user asks "Who is the chief executive?", the model might fail to answer. It learned the pattern for the exact word "CEO", but didn't "understand" the fact well enough to generalise it.

Furthermore, if a fact changes tomorrow, you would have to completely retrain the model to update its knowledge.

**Retrieval-Augmented Generation** (RAG) takes a different route. Instead of baking knowledge into the model's weights, you store your documents in a separate database; when a user asks a question, the most relevant documents are retrieved and pasted directly into the LLM's prompt — within its context window%. 

> [!info] How ChatGPT knows the news
> This is exactly how proprietary chatbots like ChatGPT or Copilot know about recent events or newly released models like Qwen3.6. They don't retrain their massive models every day. Instead, they use a form of RAG under the hood: when you ask about the news, they quietly perform a web search (retrieval), paste the articles into their hidden system prompt, and use that context to answer you.

[Chapter 8](8-rag) covers the full pipeline, the optimisation techniques, and a detailed guide for choosing between RAG and fine-tuning.

## RAG vs. fine-tuning: how to choose

- **Start with prompting** if the model already has the knowledge and just needs steering — it costs nothing to try. But if you must customise further, how do you choose?

The textbook analogy captures the core difference. **RAG is giving the student the textbook during the exam** — open-book: if the answer is in there and they can find the right page, they will answer correctly, even about material they never studied. **Fine-tuning is having the student study the textbook at home** — they internalise the general patterns and the style, but may misremember specifics, and if the textbook is revised, they must study again.

That analogy unpacks into a practical decision list:

| Your situation | Reach for |
| :--- | :--- |
| The model's *behaviour* is wrong: tone, format, terminology, a proprietary language | **Fine-tuning** |
| You need a skill no context snippet can teach | **Fine-tuning** |
| The relevant knowledge is diffuse "know-how" spread across everything, not lookup-able facts | **Fine-tuning** |
| The model gets *facts* wrong, or doesn't know your domain's content | **RAG** |
| Your data changes daily/weekly | **RAG** — update the index, not the model |
| Users must be able to verify answers against sources | **RAG** — citations come almost for free |
| Both: your domain's *style* and its *fresh facts* | **Both** — a LoRA%-tuned model backed by RAG |

- **RAG and fine-tuning are not rivals.** They fix different failure modes and combine well; the common production pattern is a lightly fine-tuned model (for domain tone and format) with RAG supplying the facts. Getting the combination right typically takes a few iterations of tuning both sides.

[👉 Working with LLMs: example scripts for fine-tuning and RAG](https://docs.csc.fi/support/tutorials/ml-llm/)

## Key takeaways

- Try prompting first; it is free. Then RAG for knowledge, fine-tuning for behaviour.
- LoRA gives most of the benefit of full fine-tuning at ~1.1–1.2× the model's memory footprint instead of 4–8×.
- RAG keeps knowledge outside the model, so updating a document never requires retraining.

## 📝 Check your knowledge

```quiz
title: Customising models

Q: Your model answers factual questions about your documents well, but its tone is wrong and it won't follow your strict report format. Your documents also update weekly. What is the best setup?
- [ ] Full fine-tuning on the documents every week.
- [ ] RAG alone — tone will fix itself with better retrieval.
- [x] A lightly fine-tuned (e.g., LoRA) model for tone and format, combined with RAG for the up-to-date facts.
- [ ] A larger context window with no retrieval.
> Fine-tuning fixes behaviour (tone, format, skills); RAG fixes knowledge (facts, freshness, citations). They address different failure modes and combine well in production.

---

Q: How does LoRA (Low-Rank Adaptation) make fine-tuning much cheaper and faster than full-parameter fine-tuning?
- [ ] It trains the model on fewer documents.
- [ ] It reduces the context window to save memory.
- [ ] It automatically translates the dataset into smaller languages.
- [x] It freezes the original model and only updates small, injected matrices alongside the layers.
> LoRA avoids updating the massive pre-trained weights. By training only small injected matrices, it slashes memory requirements while achieving nearly identical performance.

---

Q: You have a large database of company documents that updates daily. What is the most efficient way to get an LLM to answer questions using this specific data?
- [ ] Continuous pre-training
- [x] Retrieval-Augmented Generation (RAG)
- [ ] LoRA (Parameter-Efficient Fine-Tuning)
- [ ] Full-parameter fine-tuning
> RAG retrieves relevant documents from a separate database and pastes them into the prompt. Fine-tuning is for teaching patterns or behaviour, not for constantly updating factual knowledge.
```

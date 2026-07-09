---
title: "8. Retrieval-Augmented Generation"
nav_order: 9
---

# Retrieval-Augmented Generation (RAG)

**Retrieval-Augmented Generation** (RAG) is one of the most popular ways of deploying LLMs on private or fast-changing data — and because a RAG system is a *pipeline* of interacting components rather than a single model, building a good one is mostly about knowing which knobs exist. This chapter walks through the standard three-stage pipeline, the most effective optimisation techniques for each stage, and how to evaluate the result.

## Why RAG exists: two kinds of memory

Pre-trained large language models have proven to be able to memorise and produce significant quantities of in-depth information. However, in their standard form, they suffer from three major limitations:
1. **They cannot revise their memory:** A model's knowledge stops at its training cutoff date.
2. **They are black boxes:** It is exceptionally challenging to interpret the exact reasoning behind their outputs.
3. **They hallucinate:** They may confidently generate false or misleading information when they don't know the answer.

These shortcomings often impede the deployment of LLMs in actual industry conditions. RAG systems improve on these weaknesses by formally splitting the system's knowledge into two kinds of memory:

- **Parametric memory:** This is the knowledge baked into the model's weights (parameters) during training. It defines the model's general comprehension and language skills, but it is static and uneditable without retraining.
- **Non-parametric memory:** This is external, modifiable knowledge—typically documents in a searchable vector database—supplied to the model at inference% time. 

RAG augments the first with the second: before answering, the system *retrieves* the most relevant pieces of your document collection and pastes them into the prompt, and the model answers grounded in that context. The payoff:
- **Fewer hallucinations** — answers are anchored to real documents.
- **Fresh knowledge** — update a document and the very next query uses the new version, no retraining.
- **Transparency** — the system can cite which documents an answer came from, so users can verify it.
- **Access control** — document-level permissions can regulate who can retrieve what.

## The three-stage pipeline

A RAG system has three distinct stages: **indexing**, **retrieval**, and **generation**.

<figure>
  <img src="./assets/RAG_workflow.jpg" alt="RAG workflow diagram" />
  <figcaption><em>Figure: A typical Retrieval-Augmented Generation (RAG) workflow.</em></figcaption>
</figure>

**1. Indexing (offline, done once per document).** 
Documents are parsed and cleaned into plain text, then split into smaller pieces called **chunks**. This is necessary because whole documents (like a 100-page manual) cannot be pasted into the prompt due to context window% limits and compute costs. Each chunk is then converted into an embedding% vector% by an **embedding model** (a small, cheap neural network). Finally, the chunks and their mathematical vectors are stored in a **vector database**.

**2. Retrieval (at query time).** 
When a user asks a question, that question is encoded into an embedding using the *same* embedding model as during indexing. The system then searches the database for chunks that match the query. There are three flavours of retrieval:
- **Dense retrieval** compares the neural embeddings using cosine similarity. It is excellent at matching *semantic meaning* (e.g., "laptop won't start" mathematically matches "computer fails to boot").
- **Sparse retrieval** matches exact keywords using vectors as long as the vocabulary (mostly zeros). It is excellent for exact terms, product codes, or specific names.
- **Hybrid retrieval** combines both methods to get the best of both worlds, and is the common default in modern systems.

**3. Generation.** 
The retrieved chunks are appended to the user's query in a new, hidden system prompt—a technique called **in-prompt augmentation** (or fusion-in-decoder). The LLM is instructed to answer the user's query using the provided context. 

## Improving each stage

RAG is a fast-moving field, but a fairly stable toolbox of advanced techniques has emerged.

### Indexing techniques

The way you cut your documents into chunks drastically affects the quality of the embeddings.
- **Fixed-length chunking** (e.g., cutting every 512 tokens or by page) is the simplest approach, but it happily cuts sentences, paragraphs, and coherent thoughts in half, diluting the semantic meaning of the chunk.
- **Semantic chunking** attempts to keep related sentences together. A *breakpoint-based chunker* compares the embeddings of every pair of adjacent sentences in a document; if the similarity drops below a certain threshold, it assumes the topic has changed and inserts a chunk split. *Clustering-based* variants go further by grouping semantically related sentences even if they are non-sequential. Semantic chunking generally beats fixed-length splitting by preserving complete thoughts.

### Retrieval techniques

- **Query classification:** Not every question needs retrieval. If the model already knows the answer (e.g., "What is 2+2?"), appending irrelevant retrieved context adds noise and *hurts* performance. Adding a small, fast classifier (like [FLARE](https://arxiv.org/abs/2305.06983), which predicts future sentences and triggers retrieval only if confidence is low) to decide "retrieve or not" both improves accuracy and cuts latency.
- **Query rewriting:** User queries are often poorly phrased for search. Prompting a fast LLM to rewrite or expand the query before retrieval is a cheap, highly effective fix.
- **HyDE (Hypothetical Document Embeddings):** Questions and answers often look mathematically different in embedding space. Instead of embedding the *question*, an LLM first generates a hypothetical *answer*—fake, possibly wrong in its facts, but shaped exactly like a real document. This fake document's embedding is then used for the search, reliably landing closer to the actual target documents in the vector database.
- **Reranking:** The initial retriever's similarity ranking is incredibly fast but somewhat crude. A second, more careful (and computationally expensive) model re-scores the top candidates against the query. Studies consistently find reranking to be one of the best value-for-compute upgrades.
- **Document repacking:** The *order* in which the retrieved chunks are pasted into the prompt matters. Options are "forward" (most relevant first), "reverse" (most relevant last, closest to the actual question), and "sides" (most relevant at both ends, weakest in the middle). The "reverse" method tends to perform best, as LLMs pay the most attention to the text immediately preceding their generation.
- **Similarity cutoff:** Setting a hard threshold to discard retrieved chunks whose similarity to the query is too low. This ensures that marginally relevant or "chopped off" text never reaches the prompt to confuse the LLM.
- **Context summarisation:** Compressing the retrieved chunks before sending them to the LLM. This can be done extractively (keeping only the most important sentences) or abstractively (fusing key points into a new summary). This reduces prompt length and GPU cost, and sometimes improves accuracy by removing noise.
- **Retriever fine-tuning:** Training the embedding model itself on your own domain data. This adapts the mathematical definition of "similarity" to what *your* users mean by relevant — one of the highest-impact (and highest-effort) optimisations.

### Generation techniques

- **Prompt engineering:** Carefully instructing the model how to use the context. Even small prompt changes measurably shift RAG accuracy. Surprisingly, research shows that adding emotional weight to the prompt (e.g., *"Make sure the answer is 100% correct because my life depends on it"*) can lead to significantly higher performance.
- **Majority voting:** Because LLM sampling is not deterministic, generating several answers to the same prompt and taking the most common one buys accuracy at the cost of extra generation compute.

However, because all pipeline components are deeply interconnected, applying these techniques requires holistic tuning—a change in your chunking strategy, for instance, might necessitate adjusting your retriever's top-*K* value or rewriting your prompt.

## Evaluating a RAG system

You cannot optimise what you cannot measure, and manual output inspection does not scale. The practical approaches for evaluating RAG include:

- **Reference-based metrics:** Metrics like [BLEU](https://doi.org/10.3115/1073083.1073135), [ROUGE](http://anthology.aclweb.org/W/W04/W04-1013.pdf), and [BERTScore](https://arxiv.org/abs/1904.09675) mathematically compare the generated answers against known-correct "gold" answers. This is useful when you have a meticulously hand-crafted test set.
- **Reference-free frameworks (like [RAGAS](https://arxiv.org/abs/2309.15217)):** Instead of needing known-correct answers, these frameworks use a strong LLM as a judge to score three things separately:
  - **Faithfulness:** Are the generated answer's claims actually grounded in the retrieved context?
  - **Answer relevance:** Does the generated response actually answer the user's specific question?
  - **Context relevance:** Was the retrieved context on-topic and useful to begin with?
  Scoring the stages separately tells you *exactly which part of the pipeline to fix* (e.g., if context relevance is low, fix your chunking; if faithfulness is low, fix your prompt).
- **Public benchmarks:** Benchmarks like [FRAMES](https://arxiv.org/abs/2409.12941) (which tests temporal multi-step reasoning) or [CRAG](https://arxiv.org/abs/2406.04744) (which tests realistic, diverse topics) give a sense of the field. However, beware of **benchmark contamination**: because LLMs are trained on massive scrapes of the internet, the benchmark questions and answers may have accidentally been included in their training data, skewing the results. Your own domain test set is always superior.

> [!note] What this means on LUMI
> A RAG pipeline adds two lightweight components next to your LLM: an embedding model (small — typically well under a billion parameters, cheap to run) and a vector database (usually CPU-side). The main GPU%-side effect is *longer prompts*: every retrieved chunk adds tokens to the prefill stage and the KV Cache%, exactly the costs described in [Chapter 6](6-running-inference). Budget context accordingly — and remember that techniques like similarity cutoff and summarisation exist precisely to keep that overhead down.

## Key takeaways

- RAG bolts editable, inspectable **non-parametric memory** onto a frozen model: retrieve relevant chunks, paste them into the prompt, answer grounded in them.
- The pipeline has three stages — **indexing** (chunk + embed + store), **retrieval** (embed query, find top-*K* similar chunks), **generation** (answer from augmented prompt) — and each stage has its own optimisation toolbox.
- Advanced techniques like semantic chunking, HyDE, reranking, and emotional prompt engineering can drastically improve accuracy.
- Evaluate the pipeline holistically. Tools like RAGAS score faithfulness, answer relevance, and context relevance separately so you know exactly which stage of the pipeline to fix.

## Check your knowledge

```quiz
title: Retrieval-Augmented Generation

Q: What is the difference between parametric and non-parametric memory in an LLM system?
- [ ] Parametric memory is stored on disk, non-parametric memory in VRAM.
- [x] Parametric memory is knowledge baked into the model's weights during training; non-parametric memory is external, editable knowledge (like a document database) supplied at inference time.
- [ ] Parametric memory holds facts, non-parametric memory holds grammar rules.
- [ ] They are two names for the same thing.
> A standard LLM only has parametric memory — frozen at training time. RAG adds non-parametric memory: external documents that can be updated and inspected without retraining.

---

Q: How does semantic chunking improve the indexing stage compared to fixed-length chunking?
- [ ] It reduces the size of the vector database by compressing the text.
- [x] By comparing the embeddings of adjacent sentences, it avoids splitting coherent thoughts and paragraphs in half.
- [ ] It automatically translates the documents into different languages before chunking.
- [ ] It converts dense embeddings into sparse keywords.
> Semantic chunking (like breakpoint-based chunking) keeps related sentences together, ensuring that the semantic meaning of the chunk isn't diluted by arbitrarily cutting thoughts in half.

---

Q: How does HyDE (Hypothetical Document Embeddings) improve retrieval?
- [ ] It removes hypothetical statements from the documents before indexing.
- [ ] It caches previously retrieved documents to reduce latency.
- [x] An LLM first generates a fake, answer-shaped document for the query, and that document's embedding — rather than the raw query's — is used for the similarity search.
- [ ] It converts dense embeddings into sparse keyword vectors.
> Questions and answers often look different in embedding space. HyDE searches with something shaped like the answer, which lands closer to the real documents — at the cost of extra generation latency.

---

Q: Why can retrieving context for every single query actually hurt a RAG system's accuracy?
- [x] When the model already knows the answer, appended context that is irrelevant acts as noise and can degrade the response.
- [ ] Retrieval permanently overwrites the model's parametric memory.
- [ ] The vector database returns documents in a random order.
- [ ] Retrieved chunks are always truncated to one sentence.
> Not every question needs retrieval. Query classification — a small check deciding whether to retrieve at all — both improves accuracy and reduces latency.

---

Q: When evaluating a RAG system using a framework like RAGAS, what does the "Faithfulness" score measure?
- [ ] Whether the LLM's response is polite and helpful.
- [ ] Whether the retrieved chunks are relevant to the user's question.
- [x] Whether the claims made in the generated answer are actually grounded in the retrieved context.
- [ ] Whether the model successfully retrieved the correct document from the database.
> Faithfulness specifically checks if the model hallucinated or if it faithfully used the provided context. If faithfulness is low, you likely need to fix your generation prompt, not your retriever.
```

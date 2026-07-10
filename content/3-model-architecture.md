---
title: "3. Inside the Architecture"
nav_order: 4
---

# Inside the architecture

Not all LLMs are built the same way, and understanding their internal mechanics is key to running them efficiently. This chapter covers the two decisions that most affect your GPU% usage: how a model organises its parameters% (dense% vs. Mixture of Experts%) and how its attention% mechanism manages memory.

## The Feed-Forward Network: The Knowledge Base

A transformer model is built out of many stacked layers (or "blocks"). Each block has two main components: an **Attention** mechanism (which moves information between different tokens) and a **Feed-Forward Network** (which processes each token individually). 

You can think of the **Feed-Forward Network (FFN)** as the model's vast, internal knowledge base. While attention is responsible for connecting the words "capital", "of", and "France" together in a sentence, it is the FFN that actually stores the learned fact that the answer is "Paris". 

The vast majority of a model's parameters (its learned weights) live inside these FFNs. Because the FFN holds so much data, passing tokens through it is incredibly computationally expensive. This leads to a critical architectural decision: how should a model organise its FFN parameters to remain efficient?

## Dense vs. Mixture of Experts (MoE)

The two main architectural approaches determine how the model uses its FFN parameters:

**Dense models.**
In a **dense** model (such as Meta's Llama 3), the FFN is one massive, unified block. Every single parameter is active for processing each individual token.

- **Slightly higher quality:** Because all parameters participate in the math of every token, dense models are best when you want to squeeze the absolute maximum reasoning quality out of the model's size (and the VRAM% it consumes).
- **Computationally expensive:** Every token generated requires the math of all parameters (e.g., all 35B parameters in a 35B dense model), making it slower to run and limiting how many simultaneous requests it can process.

**Mixture of Experts (MoE).**
In a **Mixture of Experts** (MoE) model (such as Qwen3.6-35B-A3B or Mixtral-8x7B), the giant FFN is split into many smaller, specialised sub-networks called "experts." Instead of activating all parameters, a built-in "router" dynamically selects only a few experts to process each individual token.

- **Specialised routing:** An MoE model will underperform a dense model with the same *total* parameter count, but it will significantly outperform a dense model with the same *active* parameter count, using a fraction of the compute. It achieves this by routing dynamically. You can think of this conceptually like routing a math question to "math experts" instead of "poetry experts" — though in reality, experts specialise at the token level (e.g., one expert might handle verbs, another handles punctuation or specific syntax patterns), rather than high-level human domains.
- **High throughput (fast):** Because only a small fraction of the parameters are active per token, the GPU has more compute power left over. This allows the model to process **far more prompts simultaneously**, making MoE ideal for large-scale workloads.
- **High memory footprint:** The full model must still fit in GPU memory (all 35B parameters of a 35B MoE model), even though the computation per token only uses a fraction of that (e.g., 3B active parameters).

**Decoding model names.** Consider a model named **Qwen3.6-35B-A3B**. That name encodes this exact MoE architecture:

- **35B** = 35 billion total parameters (the size that must fit in GPU memory).
- **A3B** = only about 3 billion **a**ctive parameters per token (the size performing the math).

> [!info] The industry standard
> It is widely speculated that almost all of the most powerful frontier models (such as OpenAI's GPT, Google's Gemini, and Anthropic's Claude) use a Mixture of Experts architecture. By packing hundreds of billions of parameters (or trillions) into massive expert networks, these companies maximise the model's total knowledge while keeping the active compute per token low enough to serve millions of simultaneous users.

## Attention: Queries, Keys, and Values

At the heart of every modern LLM is a mechanism called **Attention**. It is how the model decides which words in your prompt and context are relevant to each other. In the sentence *"The cat sat on the mat because it was tired"*, the attention mechanism helps the model figure out that "it" refers to "the cat" — not "the mat" — by computing relationships between all tokens simultaneously.

To process these relationships, the model runs many independent attention "viewpoints" in parallel, called **attention heads** — each specialising in a different kind of relationship (such as grammar, logical structure, or subject-verb pairing).

But how does a token actually "look" at other tokens? It uses a system of **Queries**, **Keys**, and **Values**. Think of it like searching a database:
- **Query (Q):** What the current token is looking for. (e.g., in our example sentence, the word "tired" asks: *"Are there any nouns% that might be tired?"*)
- **Key (K):** The searchable label on every other token. (e.g., the word "cat" broadcasts: *"I am a noun, and I am the subject%."*)
- **Value (V):** The actual underlying meaning of the token that gets transferred.

When the model processes a token, it matches that token's **Query** against the **Keys** of all previous tokens. When there's a strong match, the **Value** of the matched token is blended into the current token, updating its meaning with that context.

### The KV cache: memory cost of attention

During text generation, the model generates responses one token at a time. To do this, the new token needs to check its Query against the Keys and Values of *all* the tokens that came before it. 

Instead of recalculating the Keys and Values for the entire conversation history on every single step, the model computes them once and stores them in the **KV Cache** (Key-Value Cache). Think of the KV Cache as the model's short-term working memory, living in GPU VRAM.

**Why does this matter on LUMI?** The KV Cache grows with every token of context and every simultaneous user, so it directly dictates how much GPU memory your model needs. If the KV Cache gets too large, you run out of VRAM and cannot serve any more concurrent users. To solve this, modern models use memory-saving architectural tricks that compress or share this cache. Much like the trade-off in MoE architectures, these tricks sacrifice a tiny bit of quality in exchange for massive efficiency, allowing the model to serve significantly more simultaneous users on the same number of GCDs. The cache's layout also constrains how you can split a model across GPUs — a practical consequence covered in [Chapter 6](6-running-inference).

> [!tip] Want to see inside the black box?
> This section only skims the surface of attention. For a step-by-step walkthrough — embeddings, Queries/Keys/Values, attention weights, multi-head attention, why the KV Cache works, and the full family of memory-saving variants (MHA, MQA, GQA, MLA) — see the next chapter, [Deep Dive: The Attention Mechanism](4-attention).

## Key takeaways

- Dense models activate everything for every token; MoE models route each token to a few experts, trading a sliver of quality for much higher throughput.
- An MoE model's *total* parameters determine its memory footprint; its *active* parameters determine its per-token compute.
- The KV Cache is the model's working memory in VRAM, and it grows with context length and concurrent users.
- Modern models use architectural tricks to shrink the KV Cache, allowing them to serve many more users per GPU.

## Check your knowledge

```quiz
title: Architecture

Q: Why can a Mixture of Experts (MoE) model process requests much faster than a dense model of the same total size?
- [ ] Because it has fewer total parameters.
- [ ] Because it has a simpler attention mechanism.
- [ ] Because it requires less VRAM to load the model weights.
- [x] Because it uses a router to only activate a small fraction of its parameters for each token.
> MoE models must still fit all their parameters in VRAM, but they dynamically activate only specific experts per token, saving massive amounts of compute time.

---

Q: During text generation, where does the model store the context of all previously generated tokens to avoid recalculating them?
- [ ] In the system's hard drive (disk).
- [ ] In the model's underlying training dataset.
- [x] In the KV Cache (Key-Value Cache) in the GPU's VRAM.
- [ ] In the CPU's L3 cache.
> The KV Cache acts as the model's short-term working memory, storing the state of past tokens in VRAM.

---

Q: Why is keeping the KV Cache in mind critical when running models on a supercomputer like LUMI?
- [x] Because it lives in VRAM and grows with context length and concurrent users, directly limiting how many requests can be processed at a time.
- [ ] Because it is the only part of the model that must be run on the CPU.
- [ ] Because it permanently modifies the model's weights during text generation.
- [ ] Because it forces you to use Dense models instead of Mixture of Experts.
> The KV Cache scales with the number of tokens and active users, making it the primary bottleneck for scaling high-throughput inference on GPU memory.
```
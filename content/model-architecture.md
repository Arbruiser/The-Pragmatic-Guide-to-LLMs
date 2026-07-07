---
title: "3. Inside the Architecture"
nav_order: 4
---

# Inside the Architecture

Not all LLMs are built the same way, and understanding their internal mechanics is key to running them efficiently. This chapter covers the two decisions that most affect your GPU usage: how a model organises its parameters% (dense vs. Mixture of Experts) and how its attention% mechanism manages memory.

## Dense vs. Mixture of Experts (MoE)

The two main architectural approaches determine how the model uses its parameters%:

**Dense models.**
In a **dense%** model (such as Meta's Llama), every single parameter is active for processing each individual token%.

- **Slightly higher quality:** Because all parameters% participate in the math of every token, dense% models are best when you want to squeeze the absolute maximum reasoning quality out of the model's size (and the VRAM it consumes).
- **Computationally expensive:** Every token generated requires the math of all parameters% (e.g., all 35B parameters% in a 35B dense% model), making it slower to run and limiting how many simultaneous requests it can process.

**Mixture of Experts (MoE).**
In a **Mixture of Experts%** (MoE) model (such as Qwen3.6-35B-A3B or Mixtral-8x7B), the network contains many specialised sub-networks called "experts." Instead of activating all parameters%, a built-in "router" dynamically selects only a few experts to process each individual token%.

- **Specialised routing:** MoE models achieve almost the same quality as dense% models of equivalent size because they route dynamically. Think of it this way: if the next token is about math, the router sends it to the math experts; it doesn't waste compute activating "poetry experts."
- **High throughput (fast):** Because only a small fraction of the parameters% are active per token, the GPU has more compute power left over. This allows the model to process **far more prompts simultaneously**, making MoE ideal for large-scale workloads.
- **High memory footprint:** The full model must still fit in GPU memory (all 35B parameters% of a 35B MoE model), even though the computation per token only uses a fraction of that (e.g., 3B active parameters%).

**Decoding model names.** Consider a model named **Qwen3.6-35B-A3B**. That name encodes this exact MoE architecture:

- **35B** = 35 billion total parameters% (the size that must fit in GPU memory).
- **A3B** = only about 3 billion **a**ctive parameters% per token (the size performing the math).

> [!info] The industry standard
> It is widely speculated that almost all of the most powerful frontier models (such as OpenAI's GPT, Google's Gemini, and Anthropic's Claude) use a Mixture of Experts% architecture. By packing hundreds of billions of parameters (or trillions) into massive expert networks, these companies maximise the model's total knowledge while keeping the active compute per token low enough to serve millions of simultaneous users.

## Attention and the KV Cache

At the heart of every modern LLM is a mechanism called **Attention%**. It is how the model decides which words in your prompt and context are relevant to each other. In the sentence *"The cat sat on the mat because it was tired"*, the attention% mechanism helps the model figure out that "it" refers to "the cat" — not "the mat" — by computing relationships between all tokens simultaneously.

To process these relationships, the model splits its attention into multiple independent "viewpoints" called **attention heads**. One head might focus on grammar, another on the subject of the sentence, another on emotional tone.

**The KV Cache: Query, Key, and Value.**
During text generation, the model stores the context of all previously processed tokens in the **KV Cache%** (Key-Value Cache) — think of it as the model's short-term working memory, living in GPU VRAM. It stores two things for each token:

- **Keys:** The token's identity (*What am I?*).
- **Values:** The token's actual content or meaning (*What do I contain?*).

When the model generates a new token (acting as the **Query** — *"What am I looking for?"*), it checks its Query against the Keys in the cache to extract the right Values. Without the cache, the model would have to recompute the entire context for every single new token.

**Types of attention architectures.**
Researchers have developed different versions of attention to optimise how the heads and the KV Cache% work together, balancing reasoning quality against GPU memory efficiency:

| Type | How it works | Memory use | Used by |
| :--- | :--- | :--- | :--- |
| **MHA** (Multi-Head Attention) | Each query head has its own separate Key and Value head. | Highest — largest KV cache | Older models (GPT-2) |
| **MQA** (Multi-Query Attention) | All query heads share a single Key and Value head. | Lowest | Some Google models |
| **GQA** (Grouped-Query Attention) | Query heads are grouped; each group shares a single Key/Value head. | Medium (the sweet spot) | Llama 3, Qwen, Mistral |

**Why does this matter on LUMI?** The attention type directly affects how much GPU memory your model uses during inference — especially with long prompts or many simultaneous users. Much like the trade-off in MoE architectures, sharing Key and Value memory across multiple Query heads (GQA or MQA) sacrifices a tiny bit of quality in exchange for massive efficiency. Because the KV Cache% takes up far less VRAM, models using GQA can serve significantly more simultaneous users on the same number of GCDs compared to older MHA models. The attention layout also constrains how you can split a model across GPUs — a practical consequence covered in the [next chapter](running-inference).

## Key takeaways

- Dense% models activate everything for every token; MoE models route each token to a few experts, trading a sliver of quality for much higher throughput%.
- An MoE model's *total* parameters determine its memory footprint; its *active* parameters determine its per-token compute.
- The KV Cache% is the model's working memory in VRAM, and it grows with context length and concurrent users.
- GQA reduces KV Cache% size dramatically, which is why modern models can serve many more users per GPU.

## 📝 Check your knowledge

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

Q: To reduce the memory footprint of the KV Cache, modern models often use Grouped-Query Attention (GQA). How does GQA achieve this?
- [ ] By compressing the context window to 512 tokens.
- [x] By making multiple Query heads share a single Key and Value head.
- [ ] By converting the prompt to 8-bit precision automatically.
- [ ] By deleting the KV Cache entirely after every sentence.
> GQA groups multiple queries to share a single Key/Value pair, significantly reducing VRAM usage compared to standard Multi-Head Attention (MHA).
```
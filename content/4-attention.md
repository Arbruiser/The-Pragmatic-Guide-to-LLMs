---
title: "4. Deep Dive: The Attention Mechanism"
nav_order: 5
---

# Deep dive: the attention mechanism

[Chapter 3](3-model-architecture) introduced attention as the mechanism that lets a model work out which tokens in the context are relevant to each other, and the KV Cache% as its memory cost. This page opens the black box: what actually happens, step by step, when a transformer "pays attention". You don't need this level of detail to size a job on LUMI — but if you want to understand *why* the KV Cache exists, why Keys and Values can be cached while Queries are thrown away, and what tricks like GQA and MLA actually do, read on.

No equations required. If you can picture a vector as an ordered list of numbers, and a matrix as a grid of them, you can follow everything here.

> [!tip] Prefer video?
> 3Blue1Brown's [visual explanation of attention](https://youtu.be/eMlx5fFNoYc) is a superb animated walkthrough of the same mechanism — highly recommended as a companion to this chapter, before or after reading.

## A tad bit of history

Before the transformer — introduced in the 2017 paper *"Attention Is All You Need"* — language models were built on recurrent networks (LSTMs and GRUs) that processed text the way a human reads: one word at a time, from left to right. To understand the 10th token, the model first had to chew through tokens 1–9 in order. This **sequential nature** made it impossible to parallelise training effectively across modern GPUs, and it caused two chronic problems:

- **Long-range forgetting:** Recurrent models maintain a single "hidden state" (a mathematical summary of everything read so far). With every new token, this single state is updated to incorporate the new meaning. Because the state has a fixed capacity, it acts as an information bottleneck — to make room for new tokens, the model is forced to overwrite or dilute earlier information, naturally giving the highest importance to the most recent tokens. By the time it reaches the end of a long passage, the beginning is often "forgotten". Attention solves this by keeping all tokens accessible: a token at the end of a 500-token paragraph can attend *directly* to the first token with zero loss of signal.
- **Slow training:** Because every token had to wait for the previous one, training could not exploit the massive parallelism of GPUs%. Attention processes all tokens at once, enabling training on enormous datasets — this is what put the "Large" in Large Language Models.

> [!note] Parallel — but only where it can be
> The parallelism applies to training and to processing your *input* (the prefill stage from [Chapter 6](6-running-inference)). Generation is still sequential: each new token depends on the one before it, which is why output appears one token at a time.

## From tokens to vectors

An LLM never sees text. As [Chapter 1](1-how-llms-work) explained, input is broken into tokens, and each token ID is swapped for its embedding — the long vector of numbers representing its meaning. Two details matter for what follows:

- **The lookup is context-free:** the token "mouse" gets the *same* vector whether the text is about computers or animals. Building context-aware meaning is precisely attention's job, as we'll see below.
- **Absolute positional encoding:** Because attention looks at all tokens simultaneously, word order would otherwise be invisible — "the dog bit the man" would look identical to "the man bit the dog". In the classic transformer design, a unique "position vector" is created for every single position from the bveginning of the input (one vector for position 1, a different vector for position 2, etc.) and added directly to the token's embedding at the beginning of the LLM architecture. While this gives the model a sense of absolute position, it cannot generalise to text lengths longer than it saw during training, because it has never learned what the unique vector for position 10,001 should look like.

> [!note] Relative distance and RoPE
> Most modern LLMs use **RoPE** (Rotary Position Embeddings), which completely replaces absolute position vectors. Instead of modifying the embedding at the start, RoPE *rotates* the Query and Key vectors inside every attention layer by an angle proportional to their position. Thanks to the mathematics of rotation, when the model calculates the match (dot product) between a rotated Query and a rotated Key, the absolute positions mathematically cancel out, and the result depends purely on the *relative distance* between the two tokens. This is intuitively similar to how humans read: we care about the distance between words ("is this adjective right next to this noun?"), not their absolute position from the start of the text. We don't process a word entirely differently just because it's the 500th word in a book rather than the first. RoPE elegantly encodes this more intuitive relative distance, allowing modern models to extrapolate to much larger context windows.

> [!info] Why LLMs are hard to interpret
> Every number in these vectors — like everything else in a transformer — is learned with a single objective: predict the next token better. The resulting values are almost completely meaningless to humans, which is a big part of why "we don't really know how LLMs work" internally.

## What attention actually does

At this point, the token's vector carries its generic dictionary meaning and its location in the sequence, but it is completely isolated from its neighbours. Attention is the mechanism that finally lets the model **move meaning from one token's vector into another's**, even across long distances. Take the phrase *"the fluffy blue creature"*: before attention, the vector for "creature" just means *some being* at position 4. After attention, the final vector for "creature" has soaked in the meanings of "fluffy" and "blue".

As introduced in [Chapter 3](3-model-architecture), the model organises this exchange using **Queries (Q)**, **Keys (K)**, and **Values (V)**. In our *"fluffy blue creature"* example, the Keys of "fluffy" and "blue" would strongly match the Query of "creature", allowing a large part of their Values to be added into the "creature" vector.

But where do **Q**, **K** and **V** actually come from? The full embedding of each token is multiplied by three learned weight matrices — **Wq**, **Wk** and **Wv**. These matrices are parameters of the model, tuned during training so that the resulting Queries, Keys and Values lead to better predictions.

## From relevance scores to attention weights

To measure how relevant one token is to another, the model takes the **dot product** of the current token's Query vector with each token's Key vector. What the dot product actually measures is whether the vectors point in the same direction. The intuition here is that if they point in a similar direction, the tokens are semantically similar and highly relevant to each other. So Queries and Keys define how important two tokens are to each other.

> [!tip] A concrete example
> Imagine the Query vector for "creature" is `[1, 0, 2]` (representing a search for something that is `[an adjective, a verb, a colour]`), and the Key vector for "blue" is `[1, 0, 2]` (it is an adjective and a colour). The dot product multiplies matching positions and adds them up: `(1×1) + (0×0) + (2×2) = 5`. A high score! But the Key for "jumped" might be `[0, 1, 0]`, yielding `(1×0) + (0×1) + (2×0) = 0`. A low score. This is how the model mathematically calculates relevance.

However, raw dot product scores can range from minus infinity to plus infinity, which is difficult for the model to work with. Two fixes turn them into usable probabilities:

1. **Scaling:** Each score is divided by the square root of the Query/Key dimensionality (length of each vector), which keeps the numbers in a numerically stable range.
2. **Softmax:** The scaled scores for the current token are passed through the softmax function, which turns them into positive weights that sum to exactly 1 — behaving like percentages of the token's attention.

The result is the **attention weight**: one number per pair of tokens, saying how much the current token should care about each other token.

## The weighted sum: blending meanings

Each attention weight acts like a volume knob. The model takes every visible token's **Value**, multiplies it by that token's attention weight, and sums everything up. The result is a single **context vector (Z)** — a summary of everything the surrounding tokens contribute to the current token's meaning.

Crucially, this includes self-reflection: the current token also matches its own Query against its own Key, deciding how much of its original identity to keep versus how much to absorb from its neighbours. **Z** is a *blend* of the token's own meaning and its context.

## Many heads, mixed together

Everything so far describes **one attention head**. Real models run many in parallel — 32 per layer is typical — and each head has its own Wq, Wk and Wv, so each learns to specialise: one head might track grammar, another which pronoun refers to which noun, another logical structure.

Each head produces its own context vector **Z**. The **Z** vectors from all heads are concatenated into one long vector and multiplied by one more learned matrix, **Wo** (the output projection), which does two jobs:

- **Cross-talk:** Without Wo, each head's findings would just sit side by side. Wo mixes the signals from all heads into one cohesive representation.
- **Learned weighting:** It lets the model amplify useful head combinations and dampen unhelpful ones.

## Residual connections and stacking blocks

The attention output represents what the *context* adds to the token — but we don't want to lose the token's original meaning. So the output is added (element-wise) back onto the vector that entered the attention block. This shortcut is called a **residual connection** (also known as a **skip connection**), and it also stabilises training by preventing vanishing gradients%.

And then everything repeats. A transformer stacks many **blocks** (or "layers"): attention → feed-forward network → next block, each with its own Wq, Wk, Wv and Wo. After every block, a token's vector becomes more context-specific. In a 32-layer model with 32 heads per layer, that is 1,024 specialised "views" of the text (32 × 32). From the second block onwards, the residual comes not from the original embedding but from the previous block's output — a vector called the **hidden state**.

## Why the KV Cache works

Now the payoff for [Chapter 3's](3-model-architecture) memory story. During generation, the model needs the Keys and Values of *every* context token to process each new token. But once a token's **K** and **V** are computed, **they never change** — they can influence future tokens, but nothing updates them. So the model computes them once and stores them in the KV Cache, instead of recomputing the whole context for every generated token.

Queries are different: a Query exists only to update the *current* token's meaning, and once that token has passed through all the layers, the Query is discarded. That is why it's a *KV* cache, not a *QKV* cache.

One subtlety: Keys and Values are fixed snapshots at a specific layer. Layer 15's Queries only ever talk to layer 15's Keys and Values — each layer keeps its own snapshot of the context.

## Self-, masked, and cross-attention

The attention we have discussed so far is called **masked self-attention** (or causal attention). It ensures that the representation of a token can only be influenced by the tokens that preceded it (Queries only look at Keys of past tokens and itself, never future tokens). This is what LLMs use to generate text.

But the mechanism comes in three flavours depending on what is allowed to attend to what:

- **Self-Attention (unmasked):** Every token sees the full sequence in both directions. This means a token's **Query** is allowed to look at the **Keys** and pull in the **Values** of "future" tokens that come *after* it in the text. This is used in *encoders*, whose job is to read an entire text that already exists and encode its full semantic meaning. Because the whole text is provided up-front, the model is free to look forward and backward to build perfect context.
- **Masked Self-Attention:** Tokens see only the past. This is used in *decoders* (like all modern LLMs) whose fundamental job is to predict the next word. The model is mathematically blocked (or "masked") from looking forward. During training, this mask prevents the model from "cheating" by looking at the answer. During generation, future tokens simply don't exist yet. (And as we'll see below, even when reading a full prompt, the model is forced to keep this mask on).
- **Cross-Attention:** Used in encoder-decoder models, for example in machine translation. If translating from Ukrainian to English, the encoder processes the Ukrainian text using unmasked self-attention. It hands the result to the decoder. The decoder then uses a cross-attention layer where the **Queries** come from the English decoder, but they are matched to the **Keys and Values** of the Ukrainian encoder! For example, while generating the English word "cat", the decoder's Query scans the encoder's Keys and strongly matches the Ukrainian token "кіт", pulling in its meaning.

> [!tip] The "Prefill" paradox: sequential rules, parallel execution
> You might be wondering: when you submit a prompt to an LLM, the whole text *is* available up-front. Does it use unmasked attention? No. The model still strictly applies the mask, because its brain was hardwired during training to only understand text by predicting the next word.
> 
> However, because the text is fully available, the GPU doesn't have to process the prompt slowly, one word after another. It computes the attention for *all* the tokens in the prompt **simultaneously in parallel** (this is the **prefill** phase). This is why an LLM can digest a 1,000-word prompt in a fraction of a second, but then takes several seconds to generate a 1,000-word response (which must physically be generated sequentially).

## Sharing and compressing: MHA, MQA, GQA and MLA

As mentioned in [Chapter 3](3-model-architecture), the specific type of attention architecture determines how much VRAM the KV Cache consumes. For context, one head looking at one token produces a vector. One head looking at all tokens produces a matrix. Many heads looking at the whole sequence produces a 3D tensor — and storing these massive KV tensors is what eats your GPU memory.

- **MHA (Multi-Head Attention):** Every single head has its own unique Wq, Wk, and Wv matrices. This means a 32-head model stores 32 sets of Keys and 32 sets of Values. The memory requirements are incredibly high, making it extremely slow, though the quality is excellent. It is rarely used in modern frontier models.
- **MQA (Multi-Query Attention):** MQA goes to the opposite extreme. Every head still keeps its own unique Query, but *all* heads share exactly one single Key head and one Value head. This drastically reduces the computational and memory overhead, but noticeably compromises the performance of the model.
- **GQA (Grouped-Query Attention):** This is the middle ground and today's mainstream choice (used in Llama 3 and Mistral). Each head generates its own Queries, but groups of heads share Keys and Values. For example, the architecture hardcodes that heads 1–8 share KV head "A", and heads 9–16 share KV head "B". It uses much less KV cache than MHA while maintaining almost identical quality.
- **MLA (Multi-Head Latent Attention):** Developed by DeepSeek, this takes a radically different approach. Instead of sharing **K**/**V** heads, MLA compresses the key and value tensors into a much smaller, lower-dimensional space (a single "Latent Vector") before storing them in the KV cache. At inference time, these compressed vectors are projected back to their original size using an "up-projection" matrix. While mathematically lossy, the model learns during training how to pack the information correctly so output isn't impacted. MLA achieves linear inference complexity and vastly reduces memory requirements even compared to MQA, while outperforming MHA in quality.

## Infinite context: Sliding windows and sinks

If a model needs to process an extremely long text, the KV Cache will eventually consume all available GPU memory. To solve this, developers use **Sliding-Window Attention**. Instead of global attention where every token accesses all previous tokens, it restricts the context so a Query only attends to, say, the most recent 1,000 tokens. Surprisingly, models can still recall information from several thousand tokens back, because information "hops forward" window by window through the layers!

However, researchers discovered a major problem: imagine a KV Cache limited to 1,000 tokens. With a standard sliding window, as token 1,001 is processed, token 1 gets pushed out to make room. But the moment token 1 disappears, the model's language generation instantly breaks down into gibberish!

Why? This brings us to **Attention Sinks**. Transformer models develop a peculiar behavior during training: they allocate a massive amount of attention to the very first token (like the `[BOS]` or Beginning of Sentence marker), even when it is totally irrelevant. 

Because the softmax function forces all attention weights to sum to exactly 1.0, a token cannot assign "zero" attention to its context. When a token doesn't find anything strongly relevant to look at, it still has to put that mathematical weight *somewhere*. The first token is semantically empty and, crucially, **universally visible** — because tokens are only allowed to look at the past, the very first token is the *only* token in the entire sequence that every single subsequent word can see. This makes it the perfect, safe "dumping ground" (or sink). If you delete this sink by sliding the window past it, the attention math completely crashes.

The solution is remarkably simple. Instead of a pure sliding window, you permanently pin the first "sink" token in the KV Cache, and use the remaining 999 slots for a rolling window of the most recent tokens. This allows the model to process a million-word document using a tiny amount of memory, running indefinitely without its attention mechanism ever breaking down!

## Key takeaways

- Attention solves the long-range forgetting of recurrent models by keeping all past tokens directly accessible.
- Each token gets a **Query** ("what am I looking for?"), a **Key** ("what do I contain?") and a **Value** (the content itself), produced by learned matrices Wq, Wk and Wv.
- Attention weights are scaled dot products of Queries and Keys, pushed through softmax so they sum to 1; they control how much of each Value flows into the current token.
- Keys and Values of past tokens never change — that is why they can be cached (the KV Cache) while Queries are used once and discarded.
- GQA shrinks the KV Cache by sharing **K**/**V** heads; MLA shrinks it further by compressing **K** and **V** into a small latent vector per token.

## Check your knowledge

```quiz
title: The attention mechanism

Q: In the attention mechanism, what is the role of a token's Query vector?
- [ ] It stores the token's content that gets passed on to other tokens.
- [x] It expresses what the current token is looking for, and is matched against other tokens' Keys.
- [ ] It encodes the token's position in the sequence.
- [ ] It is a compressed copy of the token kept in the KV Cache.
> The Query is the current token's question ("what am I looking for?"). It is compared against Keys via dot products; the Values of well-matching tokens are then blended into the current token.

---

Q: Why can Keys and Values be stored in the KV Cache, while Queries are not?
- [ ] Queries are too large to store in VRAM.
- [ ] Keys and Values are identical for all tokens, so only one copy is needed.
- [x] Keys and Values never change once computed, but a Query is only used once — to update its own token — and is then discarded.
- [ ] Queries are recomputed by the CPU instead of the GPU.
> Past tokens' **K** and **V** stay fixed forever (they influence future tokens but are never updated), so caching them saves recomputation. A Query serves only the current token and is thrown away afterwards.

---

Q: How does the model turn raw Query–Key relevance scores into attention weights?
- [ ] It rounds each score to the nearest whole number.
- [ ] It keeps only the single highest score and sets the rest to zero.
- [x] It scales the scores for numerical stability, then applies softmax so they are positive and sum to 1.
- [ ] It multiplies the scores by the Value vectors directly.
> The dot products are divided by the square root of the Query/Key dimension, then softmax turns them into probability-like weights that sum to 1 — the "volume knobs" for the weighted sum of Values.

---

Q: What is the key difference between GQA and MLA as KV-memory optimisations?
- [ ] GQA removes the KV Cache entirely, while MLA doubles it.
- [x] GQA makes groups of heads share Key/Value heads, while MLA compresses Keys and Values into a small latent vector that is decompressed when needed.
- [ ] GQA only works on AMD GPUs, while MLA only works on NVIDIA GPUs.
- [ ] MLA shares Queries across heads, while GQA shares nothing.
> GQA saves memory by sharing (fewer **K**/**V** heads); MLA saves memory by compression (one small latent vector per token, up-projected at use time). Both dramatically shrink the KV Cache.

---

Q: Why do transformer models "dump" excess attention on the first token of the sequence (the attention-sink effect)?
- [ ] Because the first token always contains the most important information.
- [ ] Because the KV Cache stores tokens in reverse order.
- [x] Because softmax forces attention weights to sum to 1, so when nothing is relevant, the leftover attention needs a harmless place to go — and the ever-visible, semantically empty first token is ideal.
- [ ] Because positional encodings are strongest at position zero.
> Softmax cannot assign zero attention everywhere. Models learn to park the mandatory leftover weight on the first token, which is visible from every position and carries no meaning that could be corrupted.
```

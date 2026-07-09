---
title: "5. Multimodal LLMs"
nav_order: 6
---

# Multimodal LLMs: from pixels to tokens

Modern models don't just read text — you can paste in a screenshot, a chart, or a photo and ask questions about it. But everything in the previous chapters described a machine that predicts *text tokens*. So how does a language model "see"? This chapter follows an image through the pipeline — from raw pixels to vectors the model can reason about — and explains the two ways of gluing vision and language together: **late fusion%** and **early fusion%**. It builds directly on the embeddings and attention machinery from [Chapter 4](4-attention).

## One language inside: vectors

Neural networks understand exactly one language: long vectors of floating-point% numbers. Text already arrives in that form — as [Chapter 1](1-how-llms-work) showed, each token ID looks up its row in a big learned **embedding matrix**. If a model has a vocabulary of 150,000 tokens and a hidden dimension of 4,096, that matrix has shape [150,000 × 4,096], and from the first layer to the last, the model *only* operates on those 4,096-dimensional vectors.

So the whole problem of multimodality boils down to one question: how do we turn pixels into vectors that live in the same space?

## How images become tokens

An image enters the system as a 2D grid of pixels, each with red, green and blue values from 0 to 255. Three steps convert it into something transformer-shaped:

1. **Normalisation:** Neural networks want small, zero-centred numbers, so raw pixel values (which naturally range from 0 to 255) are rescaled into a continuous range like −1.0 to +1.0. This is just simple math: dividing the pixel by 255 shrinks it to a 0.0 to 1.0 scale, and then subtracting 0.5 and multiplying by 2 shifts it to sit perfectly across zero.
2. **Patching:** Processing every individual pixel would be computationally impossible (attention math grows quadratically, so a 1024×1024 image would require over a *trillion* comparisons per layer if every pixel were a token). Instead, the image is chopped into a grid of small squares called **patches** — the visual equivalent of tokens. A 336 × 336 image cut into 14 × 14-pixel patches yields a far more manageable **576 patches**.
3. **Flattening and projection:** Each patch is flattened into a plain list of numbers (14 × 14 pixels × 3 colours = 588 values), and a quick learned matrix multiplication converts it into an embedding% vector (e.g., 768-dimensional).

The result: the image is now a sequence of 576 embedding vectors — ready for a transformer.

## The Vision Transformer (ViT)

Each patch vector so far means very little on its own: a patch of brown pixels means nothing until it knows it is surrounded by other brown patches shaping a dog's ear. Giving patches context is the job of the **Vision Transformer%** (ViT) — a stack of attention layers, exactly like the ones in [Chapter 4](4-attention), but with two differences:

**Attention is unmasked.** LLM generation is strictly forward-predictive — token #5 must not peek at token #6, so attention is masked. But all parts of an image exist *simultaneously*: the top-left corner must be able to "see" the bottom-right corner to understand the scene. A ViT therefore uses unmasked (bidirectional) self-attention, letting all 576 patches communicate freely.

**The compute wall is quadratic.** Because self-attention compares every patch against every other patch, cost grows with the *square* of the patch count: 576 patches already means 576 × 576 = 331,776 comparisons per layer. A high-resolution 1024 × 1024 image jumps to over 28 million comparisons per layer, which rapidly exhausts GPU memory — this is why some models (like DeepSeek-OCR) use "downsamplers" to merge neighboring patches together and shrink the total patch count *before* running the expensive global attention math.

After the full stack of layers, the output is still 576 vectors, but they have been profoundly transformed by all that attention mixing. When a single patch of brown pixels enters layer 1, its vector only knows "I am a brown patch". But as it passes through the layers, it constantly looks around and pulls in context from the other patches (like the floppy fur above it, and the dog's snout below it). By the time that same patch exits layer 24, its vector no longer represents raw colours — it has absorbed enough global context to represent the high-level semantic concept: "I am the ear of a Golden Retriever".

## Late fusion: the "Frankenstein" approach

Training a huge multimodal model from scratch is enormously expensive. Historically, researchers instead used **compositional** (late fusion) architectures: bolt a pre-trained vision encoder onto a pre-trained LLM, and teach them to talk to each other.

Two pieces make this work:

- **CLIP-style alignment.** A standard vision encoder groups images purely by visual similarity (yellow tennis balls end up near yellow lemons). Encoders like **CLIP** (Contrastive Language-Image Pretraining) fix this by training on huge sets of image–caption pairs. This forces the vision encoder to actually understand semantics, grouping a picture of a dog together with the *concept* of a dog inside its own mathematical space.
- **The projector.** Because late fusion bolts together independently-trained models, they don't speak the same language or have the same dimensions. A small trainable network — the **projector** (or adapter) — acts as the translator. It performs **structural matching** (upscaling the vision encoder's 768-D vectors to 4,096-D so the LLM's internal math doesn't crash) and **semantic translation**: it maps CLIP's concept of a "dog" directly to the LLM's specific text-based concept of a "dog" in its latent space%. Because CLIP already did the hard work of identifying the visual concepts, the projector just has to act as a simple dictionary between the two spaces.

The projected visual tokens are then simply inserted into the LLM's input sequence, alongside the text tokens.

> [!info] The masking paradox, resolved
> Once inserted into the LLM, those 576 visual tokens fall under the LLM's strict causal mask (where a token can only look backwards). Doesn't this break the image by preventing the first patch from seeing the last patch? No, for two reasons:
> 1. **The ViT already mixed them:** The Vision Transformer ran unmasked attention *before* the tokens ever entered the LLM. Visual token #1 already contains information that it's an ear of a retriever, not just brown fur.
> 2. **Text comes after the image:** When the LLM is *generating* tokens, they are placed *after* the image in the sequence. Because the causal mask lets a token look at everything *before* it, the generated text can freely look backwards at all 576 visual tokens simultaneously.

## Early fusion: natively multimodal models

The ViT-plus-projector approach was a stepping stone, and it has a blind spot: the image is processed in *isolation*, without knowing what the user asked. If the question is *"What's wrong with the tire?"*, an isolated vision encoder happily wastes compute analysing the sunset in the background.

State-of-the-art "omni" models discard the separate vision tower and go **early fusion** — natively multimodal. Meta's Chameleon demonstrates this openly; Google's Gemini and OpenAI's GPT-4o are widely believed to work the same way, though their exact internals are not public. In open early-fusion models, three ingredients make it work:

1. **Direct patch ingestion:** "Raw" image patches (flattened patches that only represent pixel colours) are fed straight into the main model from layer 1 — no separate ViT, no projector.
2. **Dynamic attention:** The attention heads% switch their masking rules on the fly: text tokens use strict causal (backwards-only) attention, but image tokens use bidirectional (unmasked) attention *among themselves*. Every patch in an image can see every other patch in that same image, though they still cannot peek at any future text tokens.
3. **Contextual processing:** Because the text prompt and the image patches share the same layers from the very beginning, the words "look at the tire" guide the network's early layers to spend their visual computation on the tire and ignore the irrelevant background.

One unified latent space, processing text and vision (and increasingly audio) as a single interconnected stream.

## Why not compress text the same way?

If a vision encoder can squeeze thousands of pixels into a few hundred vectors, why don't we pool text tokens just as aggressively to save compute? Because the two carry very different information density:

- **Images are redundant:** 100 pixels of blue sky compress into one token without losing meaning.
- **Text is dense:** Every single word carries highly concentrated meaning. If you mathematically merge a dozen text tokens into a single vector, you risk blurring away crucial information. Text simply lacks the "empty space" needed for aggressive compression.

The emerging twist is the reverse: using *vision* to compress *text*. Rendering a complex PDF as an image lets a vision encoder capture both the text and its **2D spatial layout** — tables, columns, charts — in far fewer tokens than the raw character stream, while a 1D text string loses that structure.

> [!note] What this means for your jobs on LUMI
> Visual input is not free. Every image you attach becomes hundreds (or thousands) of extra tokens in the context, which grow the prefill work and the KV Cache% exactly as described in [Chapter 6](6-running-inference) — so vision workloads need proportionally more VRAM headroom per request than plain text.

## Key takeaways

- Inside the model there is only one language: vectors. Multimodality means converting pixels into vectors in the same latent space as text.
- Images are normalised, chopped into **patches** (e.g., 576 for a 336 × 336 image), and projected into embedding vectors — visual tokens.
- A Vision Transformer contextualises patches with *unmasked* self-attention, and its quadratic cost is why high-resolution images are expensive.
- **Late fusion** bolts a pre-trained vision encoder (like CLIP) onto an LLM via a small **projector** network; **early fusion** feeds raw patches into one natively multimodal model that switches attention masking by token type.
- Text resists the aggressive compression that works on images — but rendering documents *as* images is an emerging way to exploit visual compression for dense text.

## Check your knowledge

```quiz
title: Multimodal LLMs

Q: What are the main steps that turn an image into transformer-ready input?
- [ ] The image is converted to a text description by OCR, then tokenised like normal text.
- [x] Pixels are normalised, the image is chopped into patches, and each patch is flattened and projected into an embedding vector.
- [ ] Each individual pixel becomes one token.
- [ ] The image file's bytes are fed directly into the attention layers.
> Normalise (0–255 → small zero-centred floats), patch (e.g., 336×336 into 576 patches of 14×14 pixels), then flatten and project each patch into an embedding vector — the visual equivalent of tokens.

---

Q: Why does a Vision Transformer use unmasked (bidirectional) attention, while LLM text generation uses masked attention?
- [ ] Because unmasked attention is cheaper to compute.
- [ ] Because images contain fewer tokens than text.
- [x] Because all parts of an image exist at once — the top-left patch must see the bottom-right — whereas text generation must not peek at future tokens.
- [ ] Because vision models do not use attention at all.
> Text generation is forward-predictive, so the causal mask prevents cheating. An image has no "future": every patch may freely attend to every other patch to understand the global scene.

---

Q: In a late-fusion multimodal model, what is the job of the projector?
- [ ] It compresses the KV Cache after every image.
- [ ] It converts the image into a JPEG before processing.
- [x] It maps the vision encoder's output vectors into the LLM's larger embedding space, translating visual concepts into the LLM's latent language.
- [ ] It decides which attention heads process the image.
> The projector (adapter) bridges the "modality gap": it rescales the vision encoder's vectors (e.g., 768-D → 4,096-D) and semantically translates visual concepts into the LLM's text-based latent space.

---

Q: What key advantage does early fusion (native multimodality) have over the ViT-plus-projector approach?
- [ ] It removes the need for GPUs when processing images.
- [x] The text prompt and image patches share the same layers from the start, so the question can steer visual processing toward the relevant parts of the image.
- [ ] It guarantees the model never hallucinates about images.
- [ ] It makes attention costs grow linearly instead of quadratically.
> In late fusion, the vision encoder analyses the image in isolation, ignoring the question. In early fusion, "look at the tire" guides the early layers to focus visual compute on the tire from the very beginning.

---

Q: Why is aggressive token compression applied to images but not to text?
- [x] Images are highly redundant, while in text a single changed word can alter the entire meaning.
- [ ] Text tokens are already smaller than image tokens in memory.
- [ ] Compression algorithms only exist for pixel data.
- [ ] Text compression is forbidden by tokeniser standards.
> 100 pixels of blue sky can collapse into one token losslessly in meaning; text is information-dense ("I am not guilty"), and pooling it risks destroying critical facts — and hampers token-by-token generation.
```

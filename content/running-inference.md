---
title: "4. Running Inference on LUMI"
nav_order: 5
---

# Running Inference on LUMI

**Inference%** is the process of using an already-trained model to generate outputs — answering questions, writing code, summarising documents. This is what happens when you load a pre-trained model and ask it to generate text. This chapter covers the practical side: where the performance bottlenecks are, how to split models across GPUs, how many GCDs to request, and which engine to use.

> [!note] LUMI hardware in one sentence
> Each AMD MI250X GPU on LUMI consists of **2 Graphics Compute Dies (GCDs)**, and each GCD acts as an independent device with its own **64 GB** of VRAM — so when sizing jobs, think in GCDs, not GPUs.

## The inference bottleneck: compute vs. memory bandwidth

To understand inference performance, distinguish three stages of data movement, which together dictate your overall **throughput%** (the rate at which the model processes and generates tokens%):

- **Initialisation (Disk → VRAM):** Model weights move from the parallel file system into GPU memory. This happens once at startup and takes several minutes. The weights must fit entirely in VRAM (assuming no offloading), and they stay there as long as the model is running.
- **Prefill stage (compute bound):** The model processes the entire input prompt (and conversation history) in parallel. Since the GPU handles all input tokens at once, the bottleneck is the hardware's raw mathematical throughput (FLOPs). Prefill throughput is how many **input** tokens the model can **process** per second.
- **Decode stage (memory bandwidth bound):** Tokens are generated one by one. For every single token produced, the GPU must reload all the model weights from VRAM into the compute cores. Performance therefore depends on **memory bandwidth** (how fast data can move), not on how fast the GPU can calculate. Decode throughput is how many **output** tokens the model can **generate** per second.

This is also why long conversations get expensive: as [Chapter 1](how-llms-work) explained, the client resends the full history on every request, so the prefill stage grows with the conversation — and so does the VRAM consumed by the KV Cache%.

## Scaling across GPUs (parallelism)

When a model is too large for a single GCD, or when you need to process massive amounts of data simultaneously, you can distribute the workload across multiple GCDs. The three main strategies are:

- **Tensor Parallelism%** (TP): Splits the *individual mathematical operations* of the model across multiple GPUs — comparable to multiple mechanics working on the same car engine simultaneously. All GPUs are active at the same time, calculating different pieces of the same token. This requires ultra-fast, constant communication between the GPUs. In vLLM%, you control this with a parameter like `tensor_parallel_size=2`.
- **Pipeline Parallelism%** (PP): Splits the model *sequentially by its layers*, like a factory assembly line. GPU 1 holds layers 1–40, GPU 2 holds layers 41–80; GPU 1 processes a token through its layers and passes the intermediate result to GPU 2. This requires much less communication but can leave GPUs idle waiting for their turn ("pipeline bubbles").
- **Data Parallelism%** (DP): Loads a *full, independent replica* of the entire model onto each GPU. Nothing is split — each replica processes completely different batches of prompts. This doesn't help if the model is too large for one GCD's VRAM, but it drastically increases how many requests your server can handle per second.

## Sizing your workload

Now the practical part: how many GCDs should you request for your model? In the standard BF16 (16-bit) precision, each parameter takes **2 bytes** of memory — roughly **2 GB of VRAM per billion parameters**, plus headroom for the KV Cache%:

| Model size | Memory required (BF16) | Recommended GCDs (64 GB each) |
| :--- | :--- | :--- |
| 7B parameters | ~14 GB | **1 GCD** |
| 35B parameters | ~70 GB | **2 GCDs** |
| 70B parameters | ~140 GB | **4 GCDs** (3 is unsupported/inefficient) |

> [!warning] The power-of-two / GQA rule
> In vLLM%, `tensor_parallel_size` must evenly divide the number of Key-Value (KV) attention heads in the model. Because many modern models use Grouped-Query Attention (GQA) with an even number of KV heads, your tensor parallel size must be divisible by 2 — meaning you can only partition across **1, 2, 4, or 8 GCDs**. Attempting to split a model across 3 GCDs will fail.

> [!tip] Planning for KV Cache headroom
> The recommended sizes above leave healthy headroom for vLLM's KV Cache% and runtime activations. If your model's weight size is very close to your memory limit (e.g., a 60B model requiring ~120 GB on 2 GCDs with 128 GB total), do not request 1 extra GCD (3 total) — the divisibility rule forbids it. Scale to the next power of two (**4 GCDs**) instead.

## Controlling the output

When running inference, you can adjust several parameters to control how the model selects the next token%, balancing predictability against creativity. Most inference engines allow you to set these per request:

- **Temperature%**: Controls the randomness of the model's predictions. A low temperature% (e.g., `0` or `0.2`) makes the model more deterministic (but not fully), almost always picking the most likely next word — ideal for coding, mathematics, or extracting facts. A high temperature% (e.g., `0.8` or `1.0`) flattens the probabilities, allowing less likely words through — better for creative writing or brainstorming.
- **Top-P%** (nucleus sampling): Limits the model's choices to a dynamic pool of the most probable tokens whose combined probabilities equal *P* (e.g., `0.9`). It cuts off the "long tail" of unlikely words while still allowing variety.
- **Top-K%**: A simpler alternative to Top-P%: the model only considers the top *K* most likely next tokens (e.g., `50`), regardless of their actual probability scores.
- **Repetition Penalty%**: Penalises tokens that have already appeared in the response, discouraging the model from repeating itself.
- **Max Tokens%**: A hard limit on the number of output tokens per response. A vital safety net that prevents runaway generations from consuming your compute allocation.

## The vLLM engine and workflows

On LUMI we typically run models with **vLLM%**. It is the recommended inference engine because it natively supports AMD GPUs and includes powerful memory and throughput optimisations — such as *PagedAttention* and *Continuous Batching* — that massively increase how many requests the hardware can handle simultaneously.

There are two main patterns for running inference on a supercomputer:

- **Offline (batch) mode:** The model loads into VRAM, processes a massive dataset of prompts all at once at maximum speed, and shuts down immediately. This is the most common pattern for supercomputer jobs and highly efficient for cluster billing.
- **Server-client mode:** The model runs as a persistent API server, allowing interactive applications (like chatbots or web interfaces) to send requests and receive real-time responses.

> [!tip] Dive deeper: the LUMI AI Guide
> To deploy these architectures yourself — including Python scripts to chat with the model interactively, secure Unix sockets for your API server, and throughput benchmarks — the **LUMI AI Guide** has a dedicated chapter with complete, copy-pasteable code examples:
>
> [👉 LUMI AI Guide: LLM Inference](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/10-LLM-inference#readme)

## Key takeaways

- Prefill (processing input) is compute bound; decode (generating output) is memory bandwidth bound.
- Rule of thumb: **~2 GB of VRAM per billion parameters** in BF16, plus KV Cache% headroom.
- Tensor parallel sizes must be powers of two on modern GQA models: 1, 2, 4, or 8 GCDs — never 3.
- Use vLLM%: offline batch mode for large datasets, server-client mode for interactive applications.

## 📝 Check your knowledge

```quiz
title: Inference on LUMI

Q: On LUMI's AMD MI250X hardware, one physical GPU consists of how many Graphics Compute Dies (GCDs)?
- [ ] 1
- [x] 2
- [ ] 4
- [ ] 8
> Every AMD MI250X GPU on LUMI contains 2 Graphics Compute Dies (GCDs), each acting as a separate device with its own 64 GB of VRAM.

---

Q: As a general rule of thumb, how much VRAM do you need to simply load a model's weights in standard 16-bit precision?
- [ ] Exactly 64 gigabytes regardless of model size.
- [ ] Roughly 1 gigabyte of VRAM per 1 billion parameters.
- [x] Roughly 2 gigabytes of VRAM per 1 billion parameters.
- [ ] It depends entirely on the length of the prompt.
> In 16-bit precision, each parameter takes up 2 bytes of memory. Therefore, an 8B model requires ~16 GB just for the weights (plus more for the KV Cache).

---

Q: In the inference process, what is the "Prefill" stage?
- [ ] Downloading the model weights from Hugging Face.
- [ ] Formatting the user's prompt with ChatML tags.
- [ ] Converting 16-bit weights to 8-bit precision.
- [x] Processing the entire input prompt in parallel before generating the first new token.
> Prefill is the compute-bound stage where the GPU processes all the input tokens simultaneously to build the initial KV Cache.

---

Q: Which hardware bottleneck limits the speed at which a model generates output tokens one-by-one (the decode phase)?
- [ ] The GPU's raw mathematical compute speed (FLOPs).
- [ ] The CPU-to-GPU PCIe transfer speed.
- [x] The GPU's memory bandwidth (how fast data moves from VRAM to compute cores).
- [ ] The speed of the parallel file system.
> During the decode phase, the GPU must reload all the model weights from VRAM to the compute cores for every single token, making memory bandwidth the primary bottleneck.

---

Q: Which parallelism strategy splits the individual mathematical operations of a model simultaneously across multiple GPUs?
- [x] Tensor Parallelism (TP)
- [ ] Pipeline Parallelism (PP)
- [ ] Context Parallelism (CP)
- [ ] Data Parallelism (DP)
> Tensor Parallelism (TP) splits the math simultaneously. Pipeline Parallelism splits layers sequentially (like an assembly line), and Data Parallelism replicates the whole model.

---

Q: Due to the divisibility rules of modern attention architectures, what is the correct way to run a model that requires ~120 GB of VRAM across LUMI's 64 GB GCDs?
- [ ] Run it on 1 GCD (64 GB).
- [ ] Run it on 3 GCDs (192 GB).
- [ ] You cannot run models larger than 64 GB on LUMI.
- [x] Run it on 4 GCDs (256 GB).
> You cannot split a model across 3 GCDs using Tensor Parallelism due to the power-of-two rule. You must scale to the next power of two (4 GCDs).

---

Q: Why is vLLM the recommended inference engine on LUMI? (select all)
- [x] It uses Continuous Batching to maximise the number of simultaneous requests.
- [ ] It requires zero configuration and works completely automatically.
- [x] It uses PagedAttention to efficiently manage KV Cache memory.
- [x] It has native, optimised support for AMD GPUs.
> vLLM is the industry standard for high-throughput inference, especially on AMD ROCm hardware, making it perfect for LUMI.

---

Q: If you need to process a massive dataset of 1 million documents as quickly as possible without needing real-time interaction, which workflow should you use?
- [ ] A purely CPU-based workflow to save GPU hours.
- [ ] Server-client mode (starting a vLLM server and making HTTP requests).
- [ ] Interactive terminal mode.
- [x] Offline (batch) inference (loading the model directly in a Python script).
> Offline batch inference is the most efficient way to maximise GPU throughput for processing massive datasets in the background.

---

Q: If you want an LLM's outputs to be highly deterministic and predictable (for example, when writing code), how should you set the Temperature parameter?
- [ ] High (e.g., 0.8 or 1.0)
- [ ] Temperature does not affect determinism.
- [x] Low (e.g., 0.0 or 0.1)
- [ ] You must set it exactly to 0.5.
> A low temperature makes the model highly deterministic by almost always picking the most mathematically likely next word.
```

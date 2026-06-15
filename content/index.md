---
title: "A technical primer on Large Language Models"
nav_order: 1
---

# A technical primer on Large Language Models

When you run your first AI job on LUMI — perhaps a Large Language Model generating text from a prompt — you might wonder: what actually happens inside those GPUs? What do LLM names like "Qwen3.6-35B-A3B" mean? Why do we use tools like "vLLM"?


## 🧠 Open-Weight LLMs on LUMI
You are likely already familiar with proprietary language models like OpenAI's GPT series or Anthropic's Claude. On LUMI, however, you have the computing power to run, modify, and fine-tune **open-weight** models.

When you hear names like **Llama 3** (by Meta), **Mistral**, or **Qwen**, these are all highly capable open-weight% alternatives. Because their underlying weights are publicly available, you are not restricted by API limits, vendor lock-in, or data privacy concerns—you can fine-tune and test them securely and privately on LUMI's hardware.

### Understanding Parameters
When you host your own models, you must understand their hardware requirements. The "Large" in LLM refers to the number of **parameters**% — the numerical values (weights) the model learned during training. Think of parameters as the model's "memory" of patterns in language. A 70-billion-parameter model has learned 70 billion individual numbers that, together, allow it to generate coherent outputs.

**Why does size matter?** Generally, more parameters mean more capability — the model can handle more nuanced tasks and retain more knowledge. But more parameters also mean more memory and more compute power to run. This is exactly why you need a supercomputer like LUMI.

**Is size everything?** While parameter count is a useful metric, it is not the only factor that determines a model's quality. AI research moves incredibly fast, and very recent models often vastly outperform older models of the exact same size due to better training data and improved architectures.

### The Benchmark Problem
Defining "model quality" is notoriously complex. There is a myriad of different benchmarks (like SWE Bench, Humanity's Last Exam, or MMMLU) that test everything from agentic coding to multilingual reasoning to data analysis. However, no single benchmark tells the whole story, and many are flawed:

- **Data Contamination:** Because LLMs are trained on internet data, the answers to benchmark tests are often accidentally included in their training data. This means the model may have "memorized" the test rather than learning to reason.
- **Over-optimization:** When a specific benchmark becomes popular, researchers optimize their models specifically to score well on it, even if the model's real-world quality suffers.
- **Subjective "Vibes":** A model might score perfectly on a standardized test but feel stiff, overly formal, or unhelpful in actual interactions. Ultimately, the only true test of model quality is running it against your specific domain use-case.

### Base vs. Instruct Models
When browsing repositories like Hugging Face, you will often see two versions of the same model: a **Base**% model and an **Instruct**% (or Chat) model. Both models ultimately have the exact same goal: to predict the most likely next token. The difference lies entirely in the data they were trained on to learn that prediction:

*   **Base Models:** These models undergo an initial, massive pre-training phase where they "read" vast amounts of raw text from the internet. Because their training data is just internet text, their only behavior is to continue a pattern. If you prompt a Base model with *"The capital of France is"*, it will correctly answer *"Paris"*. But if you prompt it with *"What is the capital of France?"*, it might just continue the pattern of questions and output *"What is the capital of Germany? What is the capital of Ukraine?"*. Base models are raw pattern-matchers, typically used as a starting point for your own fine-tuning.
*   **Instruct (Chat) Models:** These are Base models that have undergone an additional training phase (called **Alignment**%) to teach them how to act as a helpful assistant. They are post-trained on highly curated datasets of question-response pairs and instruction-following examples. Modern models usually achieve this through a technique called **DPO**% (Direct Preference Optimization). In DPO, the model is trained on datasets where humans have evaluated two possible answers to a question; the model is then mathematically rewarded for generating the human-preferred answer and penalized for the rejected one. Unless you are intentionally fine-tuning a model from scratch, you should almost always download the Instruct version.


## 🏗️ Model Architectures: Dense vs. Mixture of Experts (MoE)
Not all LLMs are built the same way. The two main architectural approaches determine how the model uses its parameters:

### Dense Models
In a **dense**% model (such as Meta's Llama), every single parameter is active for processing each individual token (word piece).

*   **Slightly higher quality:** Because all parameters participate in the math of every single token, dense models are best when you need that little extra performance gain and want to squeeze the absolute maximum reasoning quality out of the model's size (and the VRAM it consumes).
*   **Computationally expensive:** Every single token generated requires the math of all parameters (e.g., all 35B parameters in a 35B dense model), making it slower to run and severely limiting how many simultaneous requests it can process.

### Mixture of Experts (MoE)
In a **Mixture of Experts**% (MoE) model (such as Qwen3.6-35B-A3B or Mixtral-8x7B), the network contains many specialized sub-networks called "experts." Instead of activating all parameters, a built-in "router" dynamically selects only a few experts to process each individual token.

*   **Specialized routing:** MoE models achieve almost the exact same quality as dense models of equivalent size because they route dynamically. For example, if the next token is about math, the router sends it to the math experts; it doesn't waste compute activating "poetry experts."
*   **High throughput (Fast):** Because only a small fraction of the model's parameters are active per token, the GPU has more compute power left over. This allows the model to process **a lot more prompts simultaneously**, making MoE incredibly fast for large-scale workloads.
*   **High memory footprint:** The full model must still fit in GPU memory (e.g., all 35B parameters of a 35B MoE model), even though the actual computation per token only uses a fraction of that (e.g., 3B active parameters).

Consider a model named **Qwen3.6-35B-A3B**. That name encodes this exact MoE architecture:
*   **35B** = 35 billion total parameters (the size that must fit in GPU memory).
*   **A3B** = only about 3 billion active parameters per token (the size performing the math).

> [!info] The Industry Standard
> It is widely speculated that almost all of the most powerful frontier models (such as OpenAI's GPT, Google's Gemini, and Anthropic's Claude) use a Mixture of Experts architecture. By packing hundreds of billions of parameters (or trillions) into massive expert networks, these companies can maximize the model's total knowledge base while keeping the active compute per token low enough to efficiently serve millions of simultaneous users.

## 👁️ Attention: How Models "Think"
At the heart of every modern LLM is a mechanism called **Attention**%. It is how the model decides which words in your prompt and context are relevant to each other.

For example, in the sentence *"The cat sat on the mat because it was tired"*, the attention mechanism helps the model figure out that "it" refers to "the cat" — not "the mat." It does this by computing relationships between all words simultaneously.

To process these complex relationships, the model splits its attention into multiple independent "viewpoints" called **Attention Heads**. One head might focus on grammar, another on the subject of the sentence, and another on the emotional tone. 

### The KV Cache: Query, Key, and Value
During text generation, the model stores the context of all previously generated tokens in what's called the **KV Cache**% (Key-Value Cache). Think of the KV cache as the model's short-term working memory. It stores two things for each token:
* **Keys:** The token's identity (*What am I?*).
* **Values:** The token's actual content or meaning (*What do I contain?*).

When the model needs to generate a new token (acting as the **Query**, or *"What am I looking for?"*), it checks its Query against the Keys in the cache to extract the right Values. 

### Types of Attention Architectures
Over the years, researchers have developed different versions of attention to optimize how these Heads and the KV Cache work together, balancing reasoning quality against GPU memory efficiency:

| Type | How It Works | Memory Use | Used By |
| :--- | :--- | :--- | :--- |
| **MHA** (Multi-Head Attention) | Each query head has its own separate Key and Value head. | Highest — largest KV cache | Older models (GPT-2) |
| **MQA** (Multi-Query Attention) | All query heads share a single Key and Value head. | Lowest | Some Google models |
| **GQA** (Grouped-Query Attention) | Query heads are grouped; each group shares a single Key/Value head. | Medium (The Sweet Spot) | Llama 3, Qwen, Mistral |

**Why does this matter on LUMI?** The attention type directly affects how much GPU memory your model uses during inference — especially with long prompts or many simultaneous users. Much like the trade-off in MoE architectures, sharing Key and Value memory across multiple Query heads (GQA or MQA) sacrifices a tiny bit of performance in exchange for massive efficiency. Because the KV cache takes up far less VRAM, models using GQA can serve significantly more simultaneous users on the same number of GCDs compared to older MHA models.


## ⚡ Inference: Running a Trained Model
**What is inference?** It is the process of using an already-trained model to generate outputs — answering questions, writing code, summarizing documents. This is what happens when you load a pre-trained model and ask it to generate text.

### Data Movement on the GPU
To understand inference performance, it is important to distinguish between two different types of data movement:

*   **Initialisation (Disk → VRAM):** Model weights move from the parallel file system into the GPU memory. This happens once at startup and takes several minutes. The weights must fit entirely in VRAM (assuming no offloading), and they stay there as long as the model is running.
*   **Inference (VRAM → GPU Cores):** During text generation (the Decode phase), the GPU must reload the model weights from VRAM into the compute cores for *every single token* generated.

### How "Memory" (Context) Works
When interacting with an LLM, it is a common misconception that the model "remembers" the conversation naturally. In reality, LLMs are **stateless**: they treat every single request as a brand new interaction. To have a continuous conversation, the client (your application or script) must send the *entire conversation history* back to the model every single time.

*   **The cost of context:** As the conversation grows longer, the input prompt becomes larger. This means the "Prefill" stage takes longer and consumes more VRAM (to store the KV Cache). While engines like vLLM make this memory usage much more efficient, it doesn't make it free.
*   **Client-side management:** The context memory must be managed by your script or application. If you restart your script, the conversation "memory" is cleared, even if the vLLM server is still running in the background.

### Controlling the Output: Generation Parameters
When running inference, you can adjust several parameters to control how the model selects the next token, balancing between predictability and creativity. Model developers usually state the optimal parameter ranges for their specific models on their Hugging Face pages. Straying too far from these optimal values (especially Temperature) can lead to unusual and unwanted behavior, such as the model getting stuck in an infinite loop of repetition.

*   **Temperature**%: Controls the randomness of the model's predictions. A low temperature (e.g., `0` or `0.2`) makes the model more deterministic (but not fully), almost always picking the most likely next word—ideal for coding, mathematics, or extracting facts. A high temperature (e.g., `0.8` or `1.0`) flattens the probabilities, allowing the model to choose less likely words, which is better for creative writing or brainstorming.
*   **Top-P**% (Nucleus Sampling): Limits the model's choices to a dynamic pool of the most probable tokens whose combined probabilities equal the *P* value (e.g., `0.9`). It cuts off the "long tail" of unlikely words while still allowing for variety.
*   **Top-K**%: A simpler alternative to Top-P. It strictly limits the model to only consider the top *K* most likely next tokens (e.g., `50`), ignoring all others regardless of their actual probability scores.
*   **Repetition Penalty**%: Applies a penalty score to tokens that have already been generated in the response, actively discouraging the model from using the same words over and over.
*   **Max Tokens**%: A hard limit on the number of output tokens the model is allowed to generate in a single response. This acts as a vital safety net to prevent infinite repetitions from consuming all your compute resources.

Most inference engines like vLLM allow you to set these parameters per-request, meaning you can serve both creative and deterministic applications from the exact same running model.

### Understanding Throughput
**Throughput** is the rate at which the model can process and generate tokens. Performance is split into two distinct stages, each bound by different hardware limits:

*   **Prefill (Compute bound):** The model processes the entire input prompt (and history) in parallel. Since the GPU handles all input tokens at once, the bottleneck is the hardware's raw mathematical throughput (FLOPs). Prefill throughput is how many **input** tokens the model can be **processing**.
*   **Decode (Memory bandwidth bound):** Tokens are generated one by one. For every single token produced, the GPU must reload all the model's weights from VRAM to the GPU's compute cores. This makes performance dependent on Memory Bandwidth (how fast data can move) rather than how fast the GPU can calculate. Decode throughput is how many **output** tokens the model can be **generating**.


### The vLLM Inference Engine
We typically use a library called **vLLM**% to run models. It is the recommended inference engine on LUMI because it natively supports AMD GPUs and includes powerful memory and throughput optimizations (such as *PagedAttention* and *Continuous Batching*) that massively increase how many requests the hardware can handle simultaneously.

### Scaling Across GPUs: Parallelism Strategies
When a model is too large for a single GPU, or when you need to process massive amounts of data simultaneously, you can distribute the workload across multiple GPUs (or GCDs in case of LUMI where 1 GPU consists of 2 GCDs). The three main strategies are:

*   **Tensor Parallelism**% (TP): Splits the *individual mathematical operations* of the model across multiple GPUs. Think of this like multiple mechanics working on the exact same car engine simultaneously. All GPUs are active at the exact same time, calculating different pieces of the same token. This requires ultra-fast, constant communication between the GPUs. In vLLM, you control this by setting a parameter like `tensor_parallel_size=2`.
*   **Pipeline Parallelism**% (PP): Splits the model *sequentially by its layers*. Think of this like a factory assembly line. For example, GPU 1 holds layers 1–40, and GPU 2 holds layers 41–80. GPU 1 processes a token through its layers and passes the intermediate result to GPU 2 to finish the job. Unlike TP, they are not working on the exact same step simultaneously. This requires much less communication between GPUs but can leave some GPUs idle waiting for their turn ("pipeline bubbles").
*   **Data Parallelism**% (DP): Loads a *full, independent replica* of the entire model onto multiple GPUs. There is no splitting of the model here. GPU 1 and GPU 2 each process completely different batches of user prompts at the same time. While this doesn't help if a single model is too large to fit in one GPU's VRAM, it drastically increases how many requests your server can handle per second.

### Inference Workflows on LUMI
There are two main architectural patterns for running inference on a supercomputer:

*   **Offline (Batch) Mode:** The model loads into VRAM, processes a massive dataset of prompts all at once at maximum speed, and shuts down immediately. This is a common pattern for supercomputer jobs, and it is highly efficient for cluster billing.
*   **Server-Client Mode:** The model runs as a persistent API server, allowing interactive applications (like chatbots or web interfaces) to send requests and receive real-time responses.

> [!tip] Dive Deeper: The LUMI AI Guide
> If you want to deploy these architectures yourself—including writing the Python scripts to chat with the model interactively, setting up secure Unix sockets for your API server, or running throughput benchmarks—the **LUMI AI Guide** contains a dedicated chapter with complete, copy-pasteable code examples:
> 
> [👉 LUMI AI Guide: LLM Inference](https://github.com/Lumi-supercomputer/LUMI-AI-Guide/tree/main/10-LLM-inference#readme)


## 🎯 Fine-Tuning: Teaching a Model New Tricks
Pre-trained models are impressive generalists, but what if you need a model that speaks your company's specific domain — legal contracts, medical reports, or proprietary product documentation? This is where **fine-tuning**% comes in.

Fine-tuning takes a pre-trained model and trains it further on your specific data. The model retains its general language abilities while learning the patterns, terminology, and style of your domain.

### Full-Parameter Fine-Tuning
This is the "brute force" approach: you update **every single weight** in the model during training.

*   **When to use:** When you have a large, high-quality dataset and need maximum performance. This approach can fundamentally alter the model's behavior.
*   **The risk (Catastrophic Forgetting):** Because you are altering the entire model, there is a risk that the model will "forget" its pre-trained general knowledge. For example, if you heavily train a general model exclusively on legal contracts, it might lose its ability to write code or converse in multiple languages.
*   **The cost:** The memory requirement is roughly **3–4× the model size** because you need to store the model weights, the gradients (how much each weight should change), and the optimizer states (the training algorithm's internal bookkeeping). Fine-tuning a 70B model this way requires enormous GPU resources (available on LUMI).

### PEFT: Parameter-Efficient Fine-Tuning
What if you could get 90% of the benefit at 10% of the cost? **PEFT** methods freeze most of the model's original weights and only train a small number of additional parameters.

The most popular PEFT technique is **LoRA**% (Low-Rank Adaptation). It works by adding small, trainable matrices alongside the model's frozen layers. During training, only these small matrices are updated — the original model remains untouched.

*   **When to use:** When you have limited compute, a smaller dataset, or need to iterate quickly. LoRA is especially powerful for experimenting with different fine-tuning strategies.
*   **Safety from Forgetting:** Because the original model weights are frozen, the model is much less likely to suffer from catastrophic forgetting. You can even train multiple different LoRA "adapters" for different tasks and hot-swap them instantly.
*   **The savings:** Memory requirements drop dramatically — roughly **1.1–1.2× the model size** instead of 3–4×. This means you can fine-tune a 70B model on far fewer GCDs than full-parameter training would require.
*   **The trade-off:** The maximum quality ceiling is slightly lower than full fine-tuning, but for many real-world applications, the difference is negligible.

### Quantization: Doing More with Less
Another popular technique to reduce resource requirements is **quantization**%, which shrinks the model by reducing the mathematical precision of its weights. You can think of it like rounding a highly precise decimal (such as `3.14159`) down to a shorter version (`3.14`). While you lose a tiny bit of detail, the resulting number takes up significantly less memory. 

In LLMs, model weights are typically stored as high-precision 16-bit numbers. Quantization mathematically compresses these into smaller formats, like 8-bit or 4-bit. While this compression degrades the model's reasoning quality, it drastically shrinks the VRAM footprint—enabling you to run massive models on fewer GCDs. Quantization can even be combined with LoRA (called QLoRA) for memory-efficient fine-tuning on a single node.

**The LUMI Hardware Caveat (Why it might not be worth it)**
While quantization saves VRAM, it is critical to understand the hardware you are running on. The AMD MI250X GPUs on LUMI are heavily optimized for 16-bit or higher floating-point math (FP16, BF16, FP32, etc.). They **do not** have native hardware acceleration for lower precision floating-point numbers (like FP8 or FP4), though integer formats (like INT8) are supported.

If you use floating-point quantized models (whether 8-bit or 4-bit), LUMI GPUs must constantly de-quantize those numbers back to 16-bit "on the fly" in order to perform the actual calculations. Because of this massive overhead, the model will generate text **much, much slower** than a standard 16-bit model. For most workloads on LUMI, the severe speed penalty of floating-point quantization makes it simply not worth the VRAM savings.

### RAG: Retrieval-Augmented Generation
If you want your model to answer questions based on your private data, fine-tuning is not always the best approach. Fine-tuning teaches the model *how* to speak or behave, but it is notoriously bad at acting as a factual database. If a document changes, you would have to completely retrain the model to update its knowledge.

**Retrieval-Augmented Generation**% (RAG) is an alternative architecture. Instead of baking the knowledge into the model's weights, you store your documents in a separate database. When a user asks a question, a search engine retrieves the most relevant documents and pastes them directly into the LLM's prompt. 

<figure>
  <img src="./assets/RAG_workflow.png" alt="RAG workflow diagram" />
  <figcaption><em>Figure: A typical Retrieval-Augmented Generation (RAG) workflow.</em></figcaption>
</figure>

**When to choose which?**
*   **Choose RAG** if your data changes frequently, you have a massive library of documents, or you need the model to strictly cite its sources to avoid hallucinations.
*   **Choose Fine-Tuning** if you need the model to learn a completely new format (like a proprietary coding language), adopt a specific corporate tone, or if the relevant context is simply too large to fit into a single prompt.

[👉 Working with LLMs: example scripts for Fine-Tuning, Quantization and RAG](https://docs.csc.fi/support/tutorials/ml-llm/)


## 📐 Sizing Your Workload for LUMI

<figure>
  <img src="./assets/lumi-sample.jpg" alt="LUMI compute node" style="width: 80%; max-width: 100%; margin: 0 auto; display: block;" />
</figure>

Now for the practical part: how do you figure out how many GCDs to request for your specific model?

### Inference Memory (Rule of Thumb)
In the standard BF16 (16-bit) precision, each parameter takes **2 bytes** of memory:

| Model Size | Memory Required (BF16) | Recommended GCDs (64 GB each) |
| :--- | :--- | :--- |
| 7B parameters | ~14 GB | **1 GCD** |
| 35B parameters | ~70 GB | **2 GCDs** |
| 70B parameters | ~140 GB | **4 GCDs** (3 is unsupported/inefficient) |

> [!warning] The Power-of-Two / GQA Rule
> In vLLM, `tensor_parallel_size` must evenly divide the number of Key-Value (KV) attention heads in the model. Because many modern models (like our Qwen3.6) use Grouped-Query Attention (GQA) with an even number of KV heads, your tensor parallel size must be divisible by 2 (meaning you can only partition across **1, 2, 4, or 8 GCDs**). Attempting to split a model across 3 GCDs will fail.

> [!tip] Planning for KV Cache Headroom
> The recommended sizes above leave healthy headroom for vLLM's KV Cache and runtime activations. If your model's weight size is very close to your memory limit (e.g., a 60B model requiring ~120 GB on 2 GCDs with 128 GB total HBM), do not request 1 extra GCD (3 total) due to the divisibility rule. Instead, scale to the next power of two (**4 GCDs**) if you require extra headroom.


## ✅ Summary Checklist
- You understand the difference between dense and MoE model architectures.
- You understand what attention mechanisms do and how GQA improves memory efficiency.
- You know why vLLM is the recommended inference engine on LUMI.
- You understand the difference between Server-Client mode and Offline (Batch) inference.
- You know the difference between full-parameter fine-tuning and PEFT/LoRA.
- You can estimate how many GCDs your model needs based on its parameter count.

## 📝 Check Your Knowledge

```quiz
title: LLM knowledge quiz

Q: What are the main advantages of running open-weight models on LUMI instead of using proprietary APIs? (select all)
- [x] You can securely fine-tune them on your own private data.
- [ ] They don't require GPUs to run efficiently.
- [x] No vendor lock-in or strict API rate limits.
- [x] Your data is not used for training new models
> Open-weight models give you full control over your data privacy and scaling capabilities on LUMI's powerful hardware.

---

Q: Which of the following best describes the difference between a Base model and an Instruct model?
- [ ] Base models use dense architectures, while Instruct models use Mixture of Experts.
- [ ] Base models can only generate images, while Instruct models generate text.
- [ ] Base models are trained on code, while Instruct models are trained on literature.
- [x] Base models learn patterns from raw internet text, while Instruct models are post-trained on curated question-response pairs.
> Instruct models undergo an alignment phase (often using DPO) to teach them how to act as helpful assistants rather than just continuing a text pattern.

---

Q: Why can standard LLM benchmarks sometimes be misleading when evaluating a model's true quality? (select all)
- [ ] They are often written in programming languages the model doesn't understand.
- [x] The benchmark answers might have been accidentally included in the model's training data (Data Contamination).
- [ ] Benchmarks can only test models with fewer than 100 billion parameters.
- [x] Researchers might over-optimize the model specifically to score high on those tests.
> Benchmarks are useful but flawed due to contamination and over-optimization. The best test is always evaluating the model against your specific domain use-case.

---

Q: Why can a Mixture of Experts (MoE) model process requests much faster than a Dense model of the same total size?
- [ ] Because it has fewer total parameters.
- [ ] Because it has a simpler attention mechanism.
- [ ] Because it requires less VRAM to load the model weights.
- [x] Because it uses a router to only activate a small fraction of its parameters for each token.
> MoE models must still fit all their parameters in VRAM, but they dynamically activate only specific experts per token, saving massive amounts of compute time.

---

Q: As a general rule of thumb, how much VRAM do you need to simply load a model's weights in standard 16-bit precision?
- [ ] Exactly 64 gigabytes regardless of model size.
- [ ] Roughly 1 gigabyte of VRAM per 1 billion parameters.
- [x] Roughly 2 gigabytes of VRAM per 1 billion parameters.
- [ ] It depends entirely on the length of the prompt.
> In 16-bit precision, each parameter takes up 2 bytes of memory. Therefore, an 8B model requires ~16 GB just for the weights (plus more for the KV Cache).

---

Q: On LUMI's AMD MI250X hardware, one physical GPU consists of how many Graphics Compute Dies (GCDs)?
- [ ] 1
- [x] 2
- [ ] 4
- [ ] 8
> Every AMD MI250X GPU on LUMI contains 2 Graphics Compute Dies (GCDs), each acting as a separate device with its own 64 GB of VRAM.

---

Q: During text generation, where does the model store the context of all previously generated tokens to avoid recalculating them?
- [ ] In the system's hard drive (Disk).
- [ ] In the model's underlying training dataset.
- [x] In the KV Cache (Key-Value Cache) in the GPU's VRAM.
- [ ] In the CPU's L3 Cache.
> The KV Cache acts as the model's short-term working memory, storing the state of past tokens in the VRAM.

---

Q: To reduce the memory footprint of the KV Cache, modern models often use Grouped-Query Attention (GQA). How does GQA achieve this?
- [ ] By compressing the context window to 512 tokens.
- [x] By making multiple Query heads share a single Key and Value head.
- [ ] By converting the prompt to 8-bit precision automatically.
- [ ] By deleting the KV Cache entirely after every sentence.
> GQA groups multiple queries to share a single Key/Value pair, significantly reducing VRAM usage compared to standard Multi-Head Attention (MHA).

---

Q: How does an LLM "remember" a conversation over multiple turns?
- [x] It doesn't natively remember; the client application must send the entire conversation history back to the model with every new request.
- [ ] It saves the conversation to a local file on the server.
- [ ] It relies on a special Recurrent Neural Network (RNN) layer built into the architecture.
- [ ] It updates its weights after every user prompt.
> LLMs are completely stateless. Your script or application must maintain and resend the context history every time.

---

Q: In the inference process, what is the "Prefill" stage?
- [ ] Downloading the model weights from Hugging Face.
- [ ] Formatting the user's prompt with ChatML tags.
- [ ] Converting 16-bit weights to 8-bit precision.
- [x] Processing the entire input prompt in parallel before generating the first new token.
> Prefill is the compute-bound stage where the GPU processes all the input tokens simultaneously to build the initial KV Cache.

---

Q: Which hardware bottleneck limits the speed at which a model generates output tokens one-by-one (the Decode phase)?
- [ ] The GPU's raw mathematical compute speed (FLOPs).
- [ ] The CPU-to-GPU PCIe transfer speed.
- [x] The GPU's Memory Bandwidth (how fast data moves from VRAM to compute cores).
- [ ] The speed of the parallel file system.
> During the decode phase, the GPU must reload all the model weights from VRAM to the compute cores for every single token, making memory bandwidth the primary bottleneck.

---

Q: Why is vLLM the recommended inference engine on LUMI? (select all)
- [x] It uses Continuous Batching to maximize the number of simultaneous requests.
- [ ] It requires zero configuration and works completely automatically.
- [x] It uses PagedAttention to efficiently manage KV Cache memory.
- [x] It has native, optimized support for AMD GPUs.
> vLLM is the industry standard for high-throughput inference, especially on AMD ROCm hardware, making it perfect for LUMI.

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

Q: If you need to process a massive dataset of 1 million documents as quickly as possible without needing real-time interaction, which workflow should you use?
- [ ] A purely CPU-based workflow to save GPU hours.
- [ ] Server-Client mode (starting a vLLM server and making HTTP requests).
- [ ] Interactive terminal mode.
- [x] Offline (Batch) inference (loading the model directly in a Python script).
> Offline batch inference is the most efficient way to maximize GPU throughput for processing massive datasets in the background.

---

Q: If you want an LLM's outputs to be highly deterministic and predictable (for example, when writing code), how should you set the Temperature parameter?
- [ ] High (e.g., 0.8 or 1.0)
- [ ] Temperature does not affect determinism.
- [x] Low (e.g., 0.0 or 0.1)
- [ ] You must set it exactly to 0.5.
> A low temperature makes the model highly deterministic by almost always picking the most mathematically likely next word.

---

Q: Which generation parameter limits the model's choices to a dynamic pool whose combined probabilities equal a specific target value?
- [ ] Repetition Penalty
- [x] Top-P (Nucleus Sampling)
- [ ] Max Tokens
- [ ] Top-K
> Top-P (Nucleus Sampling) dynamically cuts off the long tail of unlikely words based on cumulative probability, whereas Top-K uses a strict, fixed number of choices.

---

Q: How does LoRA (Low-Rank Adaptation) make fine-tuning much cheaper and faster than full-parameter fine-tuning?
- [ ] It trains the model on fewer documents.
- [ ] It reduces the context window to save memory.
- [ ] It automatically translates the dataset into smaller languages.
- [x] It freezes the original model weights and only trains small, additional matrices alongside them.
> LoRA is a PEFT technique that achieves great results by training only a tiny fraction of new parameters, drastically reducing memory and compute requirements.

---

Q: What is a known performance caveat regarding Quantization (specifically 8-bit / FP8) on LUMI's AMD MI250X GPUs?
- [ ] The hardware cannot run quantized models at all.
- [ ] It prevents the model from generating text in English.
- [x] The MI250X lacks native hardware support for FP8 math, causing FP8 models to run significantly slower than standard 16-bit models.
- [ ] Quantization increases the VRAM requirement of the model.
> While quantization saves VRAM, the lack of native FP8/INT8 math units on the MI250X means the GPU must emulate the math, significantly reducing processing speed.

---

Q: You have a large database of company documents that updates daily. What is the most efficient way to get an LLM to answer questions using this specific data?
- [ ] Continuous Pre-training
- [x] Retrieval-Augmented Generation (RAG)
- [ ] LoRA (Parameter-Efficient Fine-Tuning)
- [ ] Full-Parameter Fine-Tuning
> RAG retrieves relevant documents from a separate database and pastes them into the prompt. Fine-tuning is for teaching patterns or behavior, not for constantly updating factual knowledge.
```

---
title: "2. Choosing a Model"
nav_order: 3
---

# Choosing a Model

You are likely already familiar with proprietary language models like OpenAI's GPT series or Anthropic's Claude. On LUMI, however, you have the computing power to run, modify, and fine-tune% **open-weight%** models — and choosing the right one is the first practical decision of any LLM project.

## Open-weight models

When you hear names like **Llama** (by Meta), **Mistral**, or **Qwen**, these are all highly capable open-weight% alternatives to proprietary models. Because their underlying weights are publicly available — most commonly via [Hugging Face](https://huggingface.co/) — you are not restricted by API limits, vendor lock-in, or data privacy concerns. You can fine-tune% and test them securely and privately on LUMI's hardware, and your data never leaves the system.

## Judging model quality

**Parameter count is a useful first filter** — a 70B model will generally outperform a 7B model from the same family — but it is far from the whole story. Recent models often vastly outperform older models of the same size thanks to better training data and improved architectures. So how do you compare models?

**The benchmark problem.** There is a myriad of benchmarks (like SWE-Bench, Humanity's Last Exam, or MMLU) that test everything from agentic coding to multilingual reasoning. However, no single benchmark tells the whole story, and many are flawed:

- **Data contamination:** Because LLMs are trained on internet data, the answers to benchmark tests are often accidentally or intentionally included in their training data. The model may have "memorised" the test rather than learned to reason.
- **Over-optimisation:** When a benchmark becomes popular, model developers optimise specifically to score well on it, even if real-world quality suffers.
- **Subjective "vibes":** A model might score perfectly on a standardised test but feel stiff, overly formal, or unhelpful in actual interactions.

Ultimately, the only true test of model quality is running it against your specific domain use-case. Benchmarks help you build a shortlist; your own evaluation picks the winner.

## Base vs. Instruct models

When browsing repositories like Hugging Face, you will often see two versions of the same model: a **Base%** model and an **Instruct%** (or Chat) model. Both have the exact same goal — predict the most likely next token% — but they differ in what they were trained on:

- **Base% models** undergo an initial, massive pre-training phase where they "read" vast amounts of raw text from the internet. Their only behaviour is to continue a pattern. If you prompt a Base% model with *"The capital of France is"*, it will correctly answer *"Paris"*. But if you prompt it with *"What is the capital of France?"*, it might just continue the pattern of questions and output *"What is the capital of Germany? What is the capital of Ukraine?"*. Base% models are raw pattern-matchers, typically used as a starting point for your own fine-tuning%.
- **Instruct% (Chat) models** are Base% models that have undergone an additional training phase (called **Alignment%**) to teach them how to act as a helpful assistant. They are post-trained on curated datasets of question-response pairs and instruction-following examples. Modern models usually achieve this through techniques like **DPO%** (Direct Preference Optimisation), where the model is trained on pairs of answers evaluated by humans and rewarded for generating the human-preferred one.

Unless you are intentionally fine-tuning% a model from scratch, you should almost always download the **Instruct** version.

> [!info] Reasoning models
> Many modern instruct models also come in a **reasoning** (or "thinking") variant, or include a switchable thinking mode. These models generate hidden intermediate tokens to "think through" a problem before answering, which improves accuracy on complex tasks like maths and coding — at the cost of generating significantly more tokens per request. Factor this in when estimating your compute budget.

## Key takeaways

- Open-weight% models give you full control: no API limits, no vendor lock-in, and your data stays on LUMI.
- Benchmarks are a starting point, not a verdict — contamination and over-optimisation make them unreliable. Test candidates on your own use-case.
- Download the **Instruct** version unless you are fine-tuning% from scratch.

## 📝 Check your knowledge

```quiz
title: Choosing a model

Q: What are the main advantages of running open-weight models on LUMI instead of using proprietary APIs? (select all)
- [x] You can securely fine-tune them on your own private data.
- [ ] They don't require GPUs to run efficiently.
- [x] No vendor lock-in or strict API rate limits.
- [x] Your data is not used for training new models.
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
- [x] The benchmark answers may have been included in the model's training data (data contamination).
- [ ] Benchmarks can only test models with fewer than 100 billion parameters.
- [x] Model developers might over-optimise specifically to score high on those tests.
> Benchmarks are useful but flawed due to contamination and over-optimisation. The best test is always evaluating the model against your specific domain use-case.
```

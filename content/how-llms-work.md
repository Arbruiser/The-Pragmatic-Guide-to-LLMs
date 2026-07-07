---
title: "1. How LLMs Work"
nav_order: 2
---

# How LLMs Work

Before diving into architectures and hardware, it's worth understanding what a Large Language Model fundamentally *does*. Everything else in this guide — memory requirements, performance bottlenecks, fine-tuning strategies — follows from a few simple mechanics described in this chapter.

## Predicting the next token

At its core, an LLM does exactly one thing: given a sequence of text, it predicts what comes next. When you send a prompt, the model computes a probability for every possible next piece of text in its vocabulary, picks one, appends it to the sequence, and repeats — one piece at a time — until the response is complete. This loop is called **autoregressive generation**, and it is why you see chatbot answers appear word by word: the model genuinely produces them that way.

Everything an LLM appears to do — answering questions, writing code, translating — emerges from this single prediction task, learned from enormous amounts of text.

## Tokens: the model's unit of text

Models don't operate on characters or whole words, but on **tokens%** — frequently occurring chunks of text. A common word like "the" is a single token, while a rarer word like "supercomputing" might be split into several (`super`, `comput`, `ing`). As a rule of thumb, one token is about ¾ of an English word, though this varies by language: text in smaller languages often splits into far more tokens than the equivalent English text.

Tokens matter practically because they are the universal unit of measurement in LLM work:

- **Performance** is measured in tokens per second (throughput%).
- **Context limits** (see below) are defined in tokens.
- **Memory usage** during generation grows with the number of tokens processed.

## Parameters: the model's learned knowledge

The "Large" in LLM refers to the number of **parameters%** — the numerical values (weights) the model learned during training. Parameters% essentially act as the model's "memory" of patterns in language. A 70-billion-parameter model has learned 70 billion individual numbers that, together, allow it to generate coherent outputs.

**Why does size matter?** Generally, more parameters mean more capability — the model can handle more nuanced tasks and retain more knowledge. But more parameters% also mean more memory and more compute power to run. This is exactly why you need a supercomputer like LUMI.

**Is size everything?** No. AI research moves incredibly fast, and very recent models often vastly outperform older models of the exact same size due to better training data and improved architectures. We return to judging model quality in the [next chapter](choosing-a-model).

## Context: why LLMs don't "remember"

When interacting with an LLM, it is a common misconception that the model "remembers" the conversation naturally. In reality, LLMs are **stateless**: they treat every single request as a brand new interaction. To have a continuous conversation, the client (your application or script) must send the *entire conversation history* back to the model every single time.

- **The context window%:** Every model has a maximum number of tokens% it can consider in a single request — its context window%. Once a conversation (plus any documents you paste in) exceeds this limit, the oldest content must be dropped or summarised.
- **Client-side management:** The context is managed by your script or application, not by the model. If you restart your script, the conversation "memory" is gone, even if the model server is still running in the background.
- **Context isn't free:** Longer context means more tokens to process on every request, which costs both time and GPU memory. Chapter 4 explains [exactly where that cost comes from](running-inference).

## Key takeaways

- LLMs generate text one token% at a time by repeatedly predicting the most likely continuation.
- Tokens% — not words — are the unit of everything: throughput, context limits, and memory.
- Parameters% are the learned weights; more parameters mean more capability but also more hardware.
- Models are stateless: your application must resend the conversation history, within the model's context window%, on every request.

## 📝 Check your knowledge

```quiz
title: How LLMs work

Q: What is the single fundamental operation an LLM performs during text generation?
- [ ] It searches a database of pre-written answers.
- [x] It repeatedly predicts the most likely next token given the sequence so far.
- [ ] It translates the prompt into a programming language and executes it.
- [ ] It retrieves the answer from the internet in real time.
> All LLM behaviour — answering, coding, translating — emerges from one loop: predict the next token, append it, repeat.

---

Q: What is a "token" in the context of LLMs?
- [ ] Exactly one word of text.
- [ ] A single character.
- [x] A frequently occurring chunk of text — roughly ¾ of an English word on average.
- [ ] A unit of GPU memory.
> Models split text into tokens: common words are one token, rare words are split into several. Tokens are the unit for throughput, context limits, and memory usage.

---

Q: How does an LLM "remember" a conversation over multiple turns?
- [x] It doesn't natively remember; the client application must send the entire conversation history back to the model with every new request.
- [ ] It saves the conversation to a local file on the server.
- [ ] It relies on a special Recurrent Neural Network (RNN) layer built into the architecture.
- [ ] It updates its weights after every user prompt.
> LLMs are completely stateless. Your script or application must maintain and resend the context history every time.
```

# Prompt Engineering with Llama 2 & 3

A curated learning repo based on the **[DeepLearning.AI x Meta course on Prompt Engineering with Llama 2 & 3](https://learn.deeplearning.ai/courses/prompt-engineering-with-llama-2)**.
This repo documents my learnings, code experiments, and prompt engineering techniques using **Llama 2 & 3** models.

🔍 To enrich the learning experience, I also explored **GPT-OSS 120B** (an open-source large model) for some sections—mainly until the **Prompt Engineering Techniques** topic. Later topics focus fully on the **Llama family** of models.

---

## 📚 Course Topics & Explanations

### 1. **Overview of Llama Model**

* **What is Llama?**
  LLaMA (Large Language Model Meta AI) is a family of foundational language models released by Meta.
  Llama 2 (2023) and Llama 3 (2024) are open-weight models trained on vast datasets and optimized for reasoning, conversation, and coding.

* **Why it matters?**
  Unlike closed-source models (like GPT-4), Llama models are **open** and allow researchers/learners to fine-tune, adapt, and deploy them for specific tasks.

* **Key takeaways:**

  * Llama 2: trained on 2T tokens, optimized for dialogue.
  * Llama 3: improved reasoning, factuality, and alignment.
  * Supports instruction-tuning, safe responses, and multi-turn dialogue.

---

### 2. **Getting Started with Llama 2 & 3**

* Setting up Llama models involves:

  * Choosing a variant: **7B, 13B, 70B** (depending on compute).
  * Using Hugging Face transformers or Meta’s provided APIs.
  * Employing quantized versions for lower hardware requirements.

* **Practical insights learned:**

  * Prompt format is key (system + user roles).
  * Llama 3 handles longer contexts better.
  * You can run smaller versions locally (7B) while bigger ones need GPUs/TPUs.

---

### 3. **Multi-turn Conversation**

* Multi-turn conversation is essential for building chatbots and assistants.

* Llama models maintain **conversation history** and handle **contextual continuity** better than many earlier open models.

* Key strategies tested:

  * Keeping track of past messages.
  * Explicitly defining "roles" (system, user, assistant).
  * Truncating or summarizing long histories for efficiency.

* **Observation with GPT-OSS 120B:**
  This model demonstrated **strong coherence in long dialogues**, but required **careful prompt formatting** compared to Llama.

---

### 4. **Prompt Engineering Techniques**

This is the **heart of the course**—learning how to craft prompts effectively.

* Covered methods:

  * **Zero-shot prompting** → just asking directly.
  * **Few-shot prompting** → showing examples first.
  * **Chain-of-thought prompting** → encouraging reasoning steps.
  * **Role prompting** → assigning persona or expertise to the model.
  * **Instruction tuning vs. in-context learning**.

* **My experiments with GPT-OSS 120B:**

  * Very effective in **few-shot settings**, especially with reasoning-heavy prompts.
  * Tended to be more **verbose** than Llama 2.
  * Required extra guardrails (safety alignment not as strong).

* **Takeaway:**
  Prompt engineering is **model-dependent**: what works for GPT-OSS may not be optimal for Llama, and vice versa.

---

### 5. **Comparing Different Llama Models**

* Llama 2 vs. Llama 3:

  * Llama 3 shows better **factual accuracy**, **reasoning depth**, and **context length**.
  * Both maintain high performance while remaining open-weight.
  * Llama 2 is still very useful for resource-constrained setups.

* Small (7B) vs. Large (70B):

  * Smaller = faster, deployable on consumer GPUs.
  * Larger = more accurate, better at nuanced reasoning.

---

### 6. **Code Llama**

* A specialized variant of Llama designed for **programming tasks**.
* Supports **Python, C++, Java, and more**.
* Can generate, debug, and explain code.
* Learners benefit by using it as a **coding assistant** without relying on closed models like GitHub Copilot.

---

### 7. **Llama Guard**

* A safety tool built around Llama for **content moderation**.
* Ensures outputs respect guidelines: prevents toxic, unsafe, or biased responses.
* Useful for deploying real-world apps responsibly.

---

### 8. **Walkthrough of Llama Helper Function (Optional)**

* The course provides helper utilities to simplify interaction with Llama.

* These functions wrap around model calls to handle:

  * Tokenization
  * Prompt formatting
  * Output parsing

* **Why it’s helpful:**
  Beginners can focus on **designing prompts** instead of worrying about the raw model API.

---

## 🧪 Why I Used **GPT-OSS 120B** (Before Switching to Llama)

For the initial topics (overview → prompt engineering), I experimented with **GPT-OSS 120B**, an open-source 120B parameter model.

**Benefits noticed:**

* Excellent **reasoning ability** and **few-shot performance**.
* Stronger results in **multi-turn conversations** compared to smaller Llama variants.
* Open-source → gives more control for experimentation compared to proprietary models.

**Limitations vs. Llama:**

* Heavier computational needs.
* Safety/alignment not as robust.
* Less optimized for instruction-following compared to Llama 2 & 3.

👉 That’s why I transitioned to **Llama** for the rest of the course.

---

## 🙏 Acknowledgements

This repo is based on the excellent **Prompt Engineering with Llama 2 & 3** course created by:

* [DeepLearning.AI](https://learn.deeplearning.ai/)
* [Meta AI](https://www.llama.com/)

Special thanks to the open-source community for **GPT-OSS 120B**, which enriched the early part of my learning journey.

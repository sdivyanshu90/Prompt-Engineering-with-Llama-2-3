# %% [markdown]
# Source notebook: L5_comparing_llama_models.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Comparing Llama Models

# %% [markdown]
# **Update: Llama 3 was released on April 18 and this notebook has been updated to compare Llama 3 and Llama 2 models hosted on Together.ai.**
# 
# - Load helper function to prompt Llama models

# %%
# Notebook cell 3
from utils import llama, llama_chat

# %% [markdown]
# ### Task 1: Sentiment Classification
# - Compare the models on few-shot prompt sentiment classification.
# - You are asking the model to return a one word response.

# %%
# Notebook cell 5
prompt = '''
Message: Hi Amit, thanks for the thoughtful birthday card!
Sentiment: Positive
Message: Hi Dad, you're 20 minutes late to my piano recital!
Sentiment: Negative
Message: Can't wait to order pizza for dinner tonight!
Sentiment: ?

Give a one word response.
'''

# %% [markdown]
# - First, use the 7B parameter chat model (`llama-2-7b-chat`) to get the response.
# 
# **Note the model names accepted by Together.ai are case insensitive and can be either "META-LLAMA/LLAMA-2-7B-CHAT-HF" or "togethercomputer/llama-2-7b-chat". The names starting with "META-LLAMA" are preferred now.**

# %%
# Notebook cell 7
response = llama(prompt,
                 #model="togethercomputer/LLama-2-7b-chat")
#                  model = "META-LLAMA/Llama-2-7B-CHAT-HF"
                model="meta-llama/Llama-3-70b-chat-hf")
print(response)

# %% [markdown]
# - Now, use the 70B parameter chat model (`llama-2-70b-chat`) on the same task

# %%
# Notebook cell 9
response = llama(prompt,
                 #model="togethercomputer/llama-2-70b-chat")
                 model="meta-llama/Llama-3-70b-chat-hf")
print(response)

# %% [markdown]
# **Using Llama 3 chat models**

# %%
# Notebook cell 11
response = llama(prompt,
                 model = "META-LLAMA/Llama-3-8B-CHAT-HF")
print(response)

# %%
# Notebook cell 12
response = llama(prompt,
                 model = "META-LLAMA/Llama-3-70B-CHAT-HF")
print(response)

# %% [markdown]
# ### Task 2: Summarization
# - Compare the models on summarization task.
# - This is the same "email" as the one you used previously in the course.

# %%
# Notebook cell 14
email = """
Dear Amit,

An increasing variety of large language models (LLMs) are open source, or close to it. The proliferation of models with relatively permissive licenses gives developers more options for building applications.

Here are some different ways to build applications based on LLMs, in increasing order of cost/complexity:

Prompting. Giving a pretrained LLM instructions lets you build a prototype in minutes or hours without a training set. Earlier this year, I saw a lot of people start experimenting with prompting, and that momentum continues unabated. Several of our short courses teach best practices for this approach.
One-shot or few-shot prompting. In addition to a prompt, giving the LLM a handful of examples of how to carry out a task — the input and the desired output — sometimes yields better results.
Fine-tuning. An LLM that has been pretrained on a lot of text can be fine-tuned to your task by training it further on a small dataset of your own. The tools for fine-tuning are maturing, making it accessible to more developers.
Pretraining. Pretraining your own LLM from scratch takes a lot of resources, so very few teams do it. In addition to general-purpose models pretrained on diverse topics, this approach has led to specialized models like BloombergGPT, which knows about finance, and Med-PaLM 2, which is focused on medicine.
For most teams, I recommend starting with prompting, since that allows you to get an application working quickly. If you’re unsatisfied with the quality of the output, ease into the more complex techniques gradually. Start one-shot or few-shot prompting with a handful of examples. If that doesn’t work well enough, perhaps use RAG (retrieval augmented generation) to further improve prompts with key information the LLM needs to generate high-quality outputs. If that still doesn’t deliver the performance you want, then try fine-tuning — but this represents a significantly greater level of complexity and may require hundreds or thousands more examples. To gain an in-depth understanding of these options, I highly recommend the course Generative AI with Large Language Models, created by AWS and DeepLearning.AI.

(Fun fact: A member of the DeepLearning.AI team has been trying to fine-tune Llama-2-7B to sound like me. I wonder if my job is at risk? 😜)

Additional complexity arises if you want to move to fine-tuning after prompting a proprietary model, such as GPT-4, that’s not available for fine-tuning. Is fine-tuning a much smaller model likely to yield superior results than prompting a larger, more capable model? The answer often depends on your application. If your goal is to change the style of an LLM’s output, then fine-tuning a smaller model can work well. However, if your application has been prompting GPT-4 to perform complex reasoning — in which GPT-4 surpasses current open models — it can be difficult to fine-tune a smaller model to deliver superior results.

Beyond choosing a development approach, it’s also necessary to choose a specific model. Smaller models require less processing power and work well for many applications, but larger models tend to have more knowledge about the world and better reasoning ability. I’ll talk about how to make this choice in a future letter.

Keep learning!

Andrew
"""

prompt = f"""
Summarize this email and extract some key points.

What did the author say about llama models?
```
{email}
```
"""

# %% [markdown]
# - First, use the 7B parameter chat model (`llama-2-7b-chat`) to summarize the email.

# %%
# Notebook cell 16
response_7b = llama(prompt,
                model="META-LLAMA/Llama-2-7B-CHAT-HF")
print(response_7b)

# %% [markdown]
# - Now, use the 13B parameter chat model (`llama-2-13b-chat`) to summarize the email.

# %%
# Notebook cell 18
response_13b = llama(prompt,
                model="META-LLAMA/Llama-2-13B-CHAT-HF")
print(response_13b)

# %% [markdown]
# - Lastly, use the 70B parameter chat model (`llama-2-70b-chat`) to summarize the email.

# %%
# Notebook cell 20
response_70b = llama(prompt,
                model="META-LLAMA/Llama-2-70B-CHAT-HF")
print(response_70b)

# %% [markdown]
# **Using Llama 3 chat models**

# %%
# Notebook cell 22
response_llama3_8b = llama(prompt,
                model="META-LLAMA/Llama-3-8B-CHAT-HF")
print(response_llama3_8b)

# %%
# Notebook cell 23
response_llama3_70b = llama(prompt,
                model="META-LLAMA/Llama-3-70B-CHAT-HF")
print(response_llama3_70b)

# %% [markdown]
# #### Model-Graded Evaluation: Summarization
# 
# - Interestingly, you can ask a LLM to evaluate the responses of other LLMs.
# - This is known as **Model-Graded Evaluation**.

# %% [markdown]
# - Create a `prompt` that will evaluate these three responses using 70B parameter chat model (`llama-2-70b-chat`).
# - In the `prompt`, provide the "email", "name of the models", and the "summary" generated by each model.

# %%
# Notebook cell 26
prompt = f"""
Given the original text denoted by `email`
and the name of several models: `model:<name of model>
as well as the summary generated by that model: `summary`

Provide an evaluation of each model's summary:
- Does it summarize the original text well?
- Does it follow the instructions of the prompt?
- Are there any other interesting characteristics of the model's output?

Then compare the models based on their evaluation \
and recommend the models that perform the best.

email: ```{email}`

model: llama-2-7b-chat
summary: {response_7b}

model: llama-2-13b-chat
summary: {response_13b}

model: llama-2-70b-chat
summary: {response_70b}

model: llama-3-8b-chat
summary: {response_llama3_8b}

model: llama-3-70b-chat
summary: {response_llama3_70b}
"""

response_eval = llama(prompt,
                model="META-LLAMA/Llama-3-70B-CHAT-HF")
print(response_eval)

# %% [markdown]
# ### Task 3: Reasoning ###
# - Compare the three models' performance on reasoning tasks.

# %%
# Notebook cell 28
context = """
Jeff and Tommy are neighbors

Tommy and Eddy are not neighbors
"""

# %%
# Notebook cell 29
query = """
Are Jeff and Eddy neighbors?
"""

# %%
# Notebook cell 30
prompt = f"""
Given this context: ```{context}```,

and the following query:
```{query}```

Please answer the questions in the query and explain your reasoning.
If there is not enough informaton to answer, please say
"I do not have enough information to answer this questions."
"""

# %% [markdown]
# - First, use the 7B parameter chat model (`llama-2-7b-chat`) for the response.

# %%
# Notebook cell 32
response_7b_chat = llama(prompt,
                        model="META-LLAMA/Llama-2-7B-CHAT-HF")
print(response_7b_chat)

# %% [markdown]
# - Now, use the 13B parameter chat model (`llama-2-13b-chat`) for the response.

# %%
# Notebook cell 34
response_13b_chat = llama(prompt,
                        model="META-LLAMA/Llama-2-13B-CHAT-HF")
print(response_13b_chat)

# %% [markdown]
# - Then, use the 70B parameter chat model (`llama-2-70b-chat`) for the response.

# %%
# Notebook cell 36
response_70b_chat = llama(prompt,
                        model="META-LLAMA/Llama-2-70B-CHAT-HF")
print(response_70b_chat)

# %% [markdown]
# - Lastly, use the Llama 3 models.

# %%
# Notebook cell 38
response_llama3_8b_chat = llama(prompt,
                        model="META-LLAMA/Llama-3-8B-CHAT-HF")
print(response_llama3_8b_chat)

# %%
# Notebook cell 39
response_llama3_70b_chat = llama(prompt,
                        model="META-LLAMA/Llama-3-70B-CHAT-HF")
print(response_llama3_70b_chat)

# %% [markdown]
# #### Model-Graded Evaluation: Reasoning
# 
# - Again, ask a LLM to compare the three responses.
# - Create a `prompt` that will evaluate these three responses using 70B parameter chat model (`llama-2-70b-chat`).
# - In the `prompt`, provide the `context`, `query`,"name of the models", and the "response" generated by each model.

# %%
# Notebook cell 41
prompt = f"""
Given the context `context:`,
Also also given the query (the task): `query:`
and given the name of several models: `mode:<name of model>,
as well as the response generated by that model: `response:`

Provide an evaluation of each model's response:
- Does it answer the query accurately?
- Does it provide a contradictory response?
- Are there any other interesting characteristics of the model's output?

Then compare the models based on their evaluation \
and recommend the models that perform the best.

context: ```{context}```

model: llama-2-7b-chat
response: ```{response_7b_chat}```

model: llama-2-13b-chat
response: ```{response_13b_chat}```

model: llama-2-70b-chat
response: ``{response_70b_chat}```

model: llama-3-8b-chat
response: ```{response_llama3_8b_chat}```

model: llama-3-70b-chat
response: ``{response_llama3_70b_chat}``
"""

# %%
# Notebook cell 42
response_eval = llama(prompt, 
                      model="META-LLAMA/Llama-3-70B-CHAT-HF")

print(response_eval)

# %%
# Notebook cell 43


# %% [markdown]
# Source notebook: L4_prompt_engineering_techniques-GPT-OSS.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

from IPython import get_ipython as _active_ipython

def get_ipython():
    """Use the active shell and expose this script namespace to line magics."""
    shell = _active_ipython()
    if shell is None:
        raise RuntimeError("This source uses IPython commands; run it with ipython instead of python.")
    import inspect
    caller = inspect.currentframe().f_back
    shell.user_ns.update(caller.f_globals)
    shell.user_ns.update(caller.f_locals)
    return shell

# %%
# Notebook cell 1
get_ipython().system('pip install --upgrade cerebras_cloud_sdk')

# %%
# Notebook cell 2
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
cerebras_api_key = user_secrets.get_secret("CEREBRAS_API")
gemini_api_key = user_secrets.get_secret("GOOGLE_API_KEY")
hf_api_key = user_secrets.get_secret("HF_API_KEY")

# %%
# Notebook cell 3
import os
import time
from cerebras.cloud.sdk import Cerebras

client = Cerebras(api_key=cerebras_api_key)

# Function to call GPT-OSS for text completion
def gpt_oss(prompt, 
            model="gpt-oss-120b", 
            temperature=0.0, 
            max_tokens=1024,
            verbose=False,
            max_tries=3):

    # Show details if verbose mode is on
    if verbose:
        print(f"Prompt:\n{prompt}\n")
        print(f"Model: {model}")
        print(f"Temperature: {temperature}")
        print(f"Max Tokens: {max_tokens}")

    # Structure the message for the API
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    # Allow multiple attempts to call the API in case of downtime or issues
    for num_tries in range(max_tries):
        try:
            # Make the API call to Cerebras GPT-OSS model
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Print the full response if verbose
            if verbose:
                print(f"Full Response Object: {chat_completion}")

            # Check if the response contains choices and return the text
            if hasattr(chat_completion, 'choices') and len(chat_completion.choices) > 0:
                response_content = chat_completion.choices[0].message.content

                # Return the formatted response in a structured manner
                formatted_response = {
                    "model": model,
                    "prompt": prompt,
                    "response": response_content,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "status": "success"
                }

                return formatted_response

            # Handle unexpected response structure
            print("Unexpected response structure:", chat_completion)
            return None

        except Exception as e:
            print(f"Error: {e}")
            print(f"Attempt {num_tries + 1}/{max_tries}")

            # Simple retry logic with exponential backoff
            wait_time = 2 ** num_tries
            print(f"Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)

    # Return failure if we exhaust the retries
    print(f"Tried {max_tries} times, but failed to get a valid response.")
    return {
        "model": model,
        "prompt": prompt,
        "response": None,
        "status": "failure",
        "error": "Exceeded max retries"
    }

# Function to handle chat-like interactions (prompt-response pairs)
def gpt_oss_chat(prompts, responses,
                 model="gpt-oss-120b", 
                 temperature=0.0, 
                 max_tokens=1024,
                 verbose=False,
                 max_tries=3):

    # Generate the prompt for chat interactions
    prompt = get_prompt_chat(prompts, responses)

    # Call gpt_oss with the generated chat prompt
    return gpt_oss(prompt, model=model, temperature=temperature, 
                   max_tokens=max_tokens, verbose=verbose, max_tries=max_tries)

# Function to build the prompt for chat interactions
def get_prompt_chat(prompts, responses):
    prompt_chat = f"<s>[INST] {prompts[0]} [/INST]"
    for n, response in enumerate(responses):
        prompt = prompts[n + 1]
        prompt_chat += f"\n{response}\n</s><s>[INST] \n{prompt}\n [/INST]"
    return prompt_chat

# %% [markdown]
# ## In-Context Learning

# %%
# Notebook cell 5
prompt = """
What is the sentiment of:
Hi Amit, thanks for the thoughtful birthday card!
"""
response = gpt_oss(prompt)
print(response['response'])

# %% [markdown]
# ### Zero-shot Prompting
# - Here is an example of zero-shot prompting.
# - You are prompting the model to see if it can infer the task from the structure of your prompt.
# - In zero-shot prompting, you only provide the structure to the model, but without any examples of the completed task.

# %%
# Notebook cell 7
prompt = """
Message: Hi Amit, thanks for the thoughtful birthday card!
Sentiment: ?
"""
response = gpt_oss(prompt)
print(response['response'])

# %% [markdown]
# ### Few-shot Prompting
# - Here is an example of few-shot prompting.
# - In few-shot prompting, you not only provide the structure to the model, but also two or more examples.
# - You are prompting the model to see if it can infer the task from the structure, as well as the examples in your prompt.

# %%
# Notebook cell 9
prompt = """
Message: Hi Dad, you're 20 minutes late to my piano recital!
Sentiment: Negative

Message: Can't wait to order pizza for dinner tonight
Sentiment: Positive

Message: Hi Amit, thanks for the thoughtful birthday card!
Sentiment: ?
"""
response = gpt_oss(prompt)
print(response['response'])

# %% [markdown]
# ### Specifying the Output Format
# - You can also specify the format in which you want the model to respond.
# - In the example below, you are asking to "give a one word response".

# %%
# Notebook cell 11
prompt = """
Message: Hi Dad, you're 20 minutes late to my piano recital!
Sentiment: Negative

Message: Can't wait to order pizza for dinner tonight
Sentiment: Positive

Message: Hi Amit, thanks for the thoughtful birthday card!
Sentiment: ?

Give a one word response.
"""
response = gpt_oss(prompt)
print(response['response'])

# %%
# Notebook cell 12
prompt = """
Message: Hi Dad, you're 20 minutes late to my piano recital!
Sentiment: Negative

Message: Can't wait to order pizza for dinner tonight
Sentiment: Positive

Message: Hi Amit, thanks for the thoughtful birthday card!
Sentiment: 

Respond with either positive, negative, or neutral.
"""
response = gpt_oss(prompt)
print(response['response'])

# %% [markdown]
# ### Role Prompting
# - Roles give context to LLMs what type of answers are desired.
# - Llama 2 often gives more consistent responses when provided with a role.
# - First, try standard prompt and see the response.

# %%
# Notebook cell 14
prompt = """
How can I answer this question from my friend:
What is the meaning of life?
"""
response = gpt_oss(prompt)
print(response['response'])

# %% [markdown]
# - Now, try it by giving the model a "role", and within the role, a "tone" using which it should respond with.

# %%
# Notebook cell 16
role = """
Your role is a life coach \
who gives advice to people about living a good life.\
You attempt to provide unbiased advice.
You respond in the tone of an English pirate.
"""

prompt = f"""
{role}
How can I answer this question from my friend:
What is the meaning of life?
"""
response = gpt_oss(prompt)
print(response['response'])

# %% [markdown]
# ### Summarization
# - Summarizing a large text is another common use case for LLMs. Let's try that!

# %%
# Notebook cell 18
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

# %%
# Notebook cell 19
prompt = f"""
Summarize this email and extract some key points.
What did the author say about llama models?:

email: {email}
"""

response = gpt_oss(prompt)
print(response['response'])

# %%
# Notebook cell 20
from IPython.core.display import display, HTML
import markdown

def render_markdown(text):
    html_output = markdown.markdown(text)
    return html_output

html_output = render_markdown(response)
print(display(HTML(html_output)))
# print(html_output)

# %%
# Notebook cell 21


# %% [markdown]
# Source notebook: L3_multi_turn_conversations-GPT-OSS.ipynb
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
# ### LLMs are Stateless

# %%
# Notebook cell 5
prompt = """
    What are fun activities I can do this weekend?
"""
response = gpt_oss(prompt)
print(response)

# %%
# Notebook cell 6
prompt_2 = """
Which of these would be good for my health?
"""
response_2 = gpt_oss(prompt_2)
print(response_2)

# %% [markdown]
# ### Constructing multi-turn prompts
# You need to provide prior prompts and responses as part of the context of each new turn in the conversation.

# %%
# Notebook cell 8
prompt_1 = """
    What are fun activities I can do this weekend?
"""
response_1 = gpt_oss(prompt_1)

# %%
# Notebook cell 9
prompt_2 = """
Which of these would be good for my health?
"""

# %%
# Notebook cell 10
chat_prompt = f"""
<s>[INST] {prompt_1} [/INST]
{response_1}
</s>
<s>[INST] {prompt_2} [/INST]
"""
print(chat_prompt)

# %%
# Notebook cell 11
response_2 = gpt_oss(chat_prompt,
                     verbose=False)
print(response_2)

# %% [markdown]
# ### Use GPT-OSS chat helper function

# %%
# Notebook cell 13
prompt_1 = """
    What are fun activities I can do this weekend?
"""
response_1 = gpt_oss(prompt_1)

# %%
# Notebook cell 14
prompt_2 = """
Which of these would be good for my health?
"""

# %%
# Notebook cell 15
prompts = [prompt_1,prompt_2]
responses = [response_1]

# %%
# Notebook cell 16
# Pass prompts and responses to llama_chat function
response_2 = gpt_oss_chat(prompts,responses,verbose=True)

# %%
# Notebook cell 17
print(response_2)

# %%
# Notebook cell 18
# replace prompt_3 with your own question!
prompt_3 = "Which of these activites would be fun with friends?"
prompts = [prompt_1, prompt_2, prompt_3]
responses = [response_1, response_2]

response_3 = gpt_oss_chat(prompts, responses, verbose=True)

print(response_3)

# %%
# Notebook cell 19


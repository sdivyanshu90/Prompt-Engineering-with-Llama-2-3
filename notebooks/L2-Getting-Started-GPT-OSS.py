# %% [markdown]
# Source notebook: L2-Getting-Started-GPT-OSS.ipynb
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
from cerebras.cloud.sdk import Cerebras
import time

# Initialize the Cerebras client
client = Cerebras(api_key=cerebras_api_key)

# Function to make the GPT-OSS call and retrieve only the content
def gpt_oss(prompt,
            model="gpt-oss-120b", 
            temperature=1.0, 
            max_tokens=1024,
            verbose=False,
            max_tries=3):

    # Structure the message for the API
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    # Show details if verbose mode is on
    if verbose:
        print(f"Prompt:\n{prompt}\n")
        print(f"Model: {model}")
        print(f"Temperature: {temperature}")
        print(f"Max Tokens: {max_tokens}")

    # Attempt to get the response from the Cerebras API
    for num_tries in range(max_tries):
        try:
            # Call the chat completions API of Cerebras
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # If verbose, print the full response for debugging
            if verbose:
                print(f"Full Response Object: {chat_completion}")

            # Check if we have choices and return just the content
            if hasattr(chat_completion, 'choices') and len(chat_completion.choices) > 0:
                # Return the content directly from the response
                return chat_completion.choices[0].message.content

            # If the structure is unexpected, print the response and handle gracefully
            print("Unexpected response structure:", chat_completion)
            return None

        except Exception as e:
            # Catch exceptions and retry if needed
            print(f"Error: {e}")
            print(f"Attempt {num_tries + 1}/{max_tries}")

            # Simple retry logic
            wait_time = 2 ** num_tries  # Exponential backoff
            print(f"Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)

    print(f"Tried {max_tries} times, but failed to get a valid response.")
    return None

# Example usage
response = gpt_oss("Why is fast inference important?", verbose=False)
print(response)

# %%
# Notebook cell 4
# define the prompt
prompt = "Help me write a birthday card for my dear friend Andrew."

# %%
# Notebook cell 5
# pass prompt to the llama function, store output as 'response' then print
response = gpt_oss(prompt)
print(response)

# %%
# Notebook cell 6
# Set verbose to True to see the full prompt that is passed to the model.
prompt = "Help me write a birthday card for my dear friend Andrew."
response = gpt_oss(prompt, verbose=True)

# %%
# Notebook cell 7
### chat model
prompt = "What is the capital of France?"
response = gpt_oss(prompt, 
                 verbose=True)

# %%
# Notebook cell 8
print(response)

# %%
# Notebook cell 9
prompt = """
Help me write a birthday card for my dear friend Andrew.
Here are details about my friend:
He likes long walks on the beach and reading in the bookstore.
His hobbies include reading research papers and speaking at conferences.
His favorite color is light blue.
He likes pandas.
"""
response = gpt_oss(prompt, temperature=0.0)
print(response)

# %%
# Notebook cell 10
# Run the code again - the output should be identical
response = gpt_oss(prompt, temperature=0.0)
print(response)

# %%
# Notebook cell 11
prompt = """
Help me write a birthday card for my dear friend Andrew.
Here are details about my friend:
He likes long walks on the beach and reading in the bookstore.
His hobbies include reading research papers and speaking at conferences.
His favorite color is light blue.
He likes pandas.
"""
response = gpt_oss(prompt, temperature=0.9)
print(response)

# %%
# Notebook cell 12
# run the code again - the output should be different
response = gpt_oss(prompt, temperature=0.9)
print(response)

# %% [markdown]
# ### Changing the max tokens setting

# %%
# Notebook cell 14
prompt = """
Help me write a birthday card for my dear friend Andrew.
Here are details about my friend:
He likes long walks on the beach and reading in the bookstore.
His hobbies include reading research papers and speaking at conferences.
His favorite color is light blue.
He likes pandas.
"""
response = gpt_oss(prompt, max_tokens=20)
print(response)

# %%
# Notebook cell 15
with open("/kaggle/input/prompt-engineering-with-gpt-oss/TheVelveteenRabbit.txt", "r", encoding='utf=8') as file:
    text = file.read()

# %%
# Notebook cell 16
prompt = f"""
Give me a summary of the following text in 50 words:\n\n
{text}
"""
response = gpt_oss(prompt)

# %%
# Notebook cell 17
print(response)

# %%
# Notebook cell 18
prompt = """
Help me write a birthday card for my dear friend Andrew.
Here are details about my friend:
He likes long walks on the beach and reading in the bookstore.
His hobbies include reading research papers and speaking at conferences.
His favorite color is light blue.
He likes pandas.
"""
response = gpt_oss(prompt)
print(response)

# %%
# Notebook cell 19
prompt_2 = """
Oh, he also likes teaching. Can you rewrite it to include that?
"""
response_2 = gpt_oss(prompt_2)
print(response_2)

# %%
# Notebook cell 20


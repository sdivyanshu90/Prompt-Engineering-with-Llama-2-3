# %% [markdown]
# Source notebook: L8_walkthrough_helper_function.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Lesson 8 (Optional): Llama Helper Function
# 
# Here, you'll walk through the code that makes up the `llama` helper function that you've been using throughout the course, to see how it works.
# 
# ### Setup instructions for using the Together.AI service outside of the classroom
# - If you wish to make API calls to together.ai on your own, outside of the classroom, you can first create an account with [Together.AI](https://api.together.xyz/).
# - You'll get an API key. The sign up process is free, and together.ai offers $25 credit for new accounts.
# - After you get the key, you can set it in your own Mac/Linux environment with
# 
# 
# ```export TOGETHER_API_KEY=<your_together_api_key>```
# or
# ```echo 'export TOGETHER_API_KEY=<your_together_api_key>' >> ~/.bashrc``` 
# - (on Windows, you can add it to your System Settings' Environment Variables).

# %%
# Notebook cell 2
# define the together.ai API url
url = "https://api.together.xyz/inference"

# %% [markdown]
# ### Optional: Python-dotenv
# - You can optionally set your API key in a text file and use python dot-env to load that API key.
# - Python-dotenv is helpful because it makes it easy to update your API keys by updating the text file.
# 
# ```
# !pip install python-dotenv
# ```
# - In the root directory of your github repo or folder that contains your jupyter notebooks, create a `.env` file.
# - Open the file and set environment variables like this:
# 
# ```
# TOGETHER_API_KEY="abc123"
# ```
# 
# Run the following dotenv functions, which will look for a `.env` file, retrieve the variables (like the `TOGETHER_API_KEY`, and load them as environment variables.

# %%
# Notebook cell 4
# Set up environment if you saved the API key in a .env file
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# %% [markdown]
# - Whether you set the environment variable with or without the `dotenv` library, you can access environment variables using the `os` (operating system) library.

# %%
# Notebook cell 6
# Set up the together.ai API key
import os
together_api_key = os.getenv('TOGETHER_API_KEY')

# %%
# Notebook cell 7
# Store keywords that will be passed to the API
headers = {
    "Authorization": f"Bearer {together_api_key}",
    "Content-Type": "application/json"}

# %%
# Notebook cell 8
# Choose the model to call
model="togethercomputer/llama-2-7b-chat"

# %%
# Notebook cell 9
prompt = """
Please write me a birthday card for my dear friend, Andrew.
"""

# %%
# Notebook cell 10
# Add instruction tags to the prompt
prompt = f"[INST]{prompt}[/INST]"
print(prompt)

# %%
# Notebook cell 11
# Set temperature and max_tokens
temperature = 0.0
max_tokens = 1024

# %%
# Notebook cell 12
data = {
    "model": model,
    "prompt": prompt,
    "temperature": temperature,
    "max_tokens": max_tokens
}
data

# %%
# Notebook cell 13
import requests
response = requests.post(url,
                         headers=headers,
                         json=data)

# %%
# Notebook cell 14
print(response)

# %%
# Notebook cell 15
response.json()

# %%
# Notebook cell 16
response.json()['output']

# %%
# Notebook cell 17
response.json()['output']['choices']

# %%
# Notebook cell 18
response.json()['output']['choices'][0]

# %%
# Notebook cell 19
response.json()['output']['choices'][0]['text']

# %% [markdown]
# ### Compare to the output of the llama helper function

# %%
# Notebook cell 21
from utils import llama

# compare to the output of the helper function
llama(prompt)

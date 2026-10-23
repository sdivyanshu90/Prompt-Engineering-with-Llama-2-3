**Notebook source files**

The source companions below preserve notebook cell order and explanations in readable code blocks. Python uses `# %%` cell markers; TypeScript uses `// %%`. The original notebooks retain their saved visualizations and course context. GitHub language statistics exclude `.ipynb` files through `.gitattributes` and count eligible implementation source normally.

Use the original notebook's environment, dependency versions, datasets, credentials and working directory. This conversion does not supply missing course helpers or fill exercise placeholders. It does not execute training, external APIs, shell commands or setup cells during validation. Review setup and execution cells before running a whole lesson.

Run ordinary Python companions with `python "path/to/lesson.py"`, or open their cells in an editor supporting Python percent scripts. Files marked **IPython** retain notebook magics and shell commands: install IPython in the lesson environment and run `ipython "path/to/lesson.py"`. Files marked **async** have an `async def main()` entry point; run the whole script to preserve its event loop. Script validation checks Python syntax, not availability or behavior of the original dependencies.

TypeScript companions preserve the Deno runtime identified by their notebooks. Run them in the original Deno environment with the permissions and dependencies required by that lesson. Repeated notebook declarations use mutable bindings and reassignment in the script. React component variants are separate `.tsx` modules in the adjacent `.components` directory; the lesson's frontend-injection helpers still require the original course environment. No Python translation is implied for TypeScript code.

| Notebook | Source companion | Execution notes |
|---|---|---|
| [notebooks/L2-Getting-Started-GPT-OSS.ipynb](notebooks/L2-Getting-Started-GPT-OSS.ipynb) | [notebooks/L2-Getting-Started-GPT-OSS.py](notebooks/L2-Getting-Started-GPT-OSS.py) | IPython |
| [notebooks/L3_multi_turn_conversations-GPT-OSS.ipynb](notebooks/L3_multi_turn_conversations-GPT-OSS.ipynb) | [notebooks/L3_multi_turn_conversations-GPT-OSS.py](notebooks/L3_multi_turn_conversations-GPT-OSS.py) | IPython |
| [notebooks/L4_prompt_engineering_techniques-GPT-OSS.ipynb](notebooks/L4_prompt_engineering_techniques-GPT-OSS.ipynb) | [notebooks/L4_prompt_engineering_techniques-GPT-OSS.py](notebooks/L4_prompt_engineering_techniques-GPT-OSS.py) | IPython |
| [notebooks/L5_comparing_llama_models.ipynb](notebooks/L5_comparing_llama_models.ipynb) | [notebooks/L5_comparing_llama_models.py](notebooks/L5_comparing_llama_models.py) | Python |
| [notebooks/L6_code_llama.ipynb](notebooks/L6_code_llama.ipynb) | [notebooks/L6_code_llama.py](notebooks/L6_code_llama.py) | Python |
| [notebooks/L7_llama_guard.ipynb](notebooks/L7_llama_guard.ipynb) | [notebooks/L7_llama_guard.py](notebooks/L7_llama_guard.py) | Python |
| [notebooks/L8_walkthrough_helper_function.ipynb](notebooks/L8_walkthrough_helper_function.ipynb) | [notebooks/L8_walkthrough_helper_function.py](notebooks/L8_walkthrough_helper_function.py) | Python |

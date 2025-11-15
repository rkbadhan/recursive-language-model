"""
Prompt templates for Recursive Language Models.

This module defines the prompts that guide the RLM's behavior, including:
- System prompts that explain the REPL environment
- User prompts that drive the iterative reasoning loop
- Examples and strategies for effective context exploration
"""

from typing import Dict, List


# Default query when none is provided
DEFAULT_QUERY = (
    "Please read through the context and answer any queries or respond to "
    "any instructions contained within it."
)


# Main system prompt explaining the RLM REPL environment
REPL_SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query(prompt)` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. A `llm_query_batch(prompts)` function for parallel LLM queries - MUCH faster than looping! Takes a list of prompts and returns a list of responses.
4. Async versions: `llm_query_async()` and `llm_query_batch_async()` for use in async code with await.
5. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.
6. Shell command execution support - you can run bash/shell commands using ```bash or ```sh code blocks.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.

Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
```repl
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {chunk}")
print(answer)
```

You can also execute shell commands using ```bash or ```sh code blocks. For example:
```bash
echo "Hello from shell"
ls -la
```

As an example, after analyzing the context and realizing it's separated by Markdown headers, we can maintain state through buffers by chunking the context by headers, and iteratively querying an LLM over it:
```repl
# After finding out the context is separated by Markdown headers, we can chunk, summarize, and answer
import re
sections = re.split(r'### (.+)', context)
buffers = []
for i in range(1, len(sections), 2):
    header = sections[i]
    info = sections[i+1]
    summary = llm_query(f"Summarize this {header} section: {info}")
    buffers.append(f"{header}: {summary}")
final_answer = llm_query(f"Based on these summaries, answer the original query: {query}\\n\\nSummaries:\\n" + "\\n".join(buffers))
```
In the next step, we can return FINAL_VAR(final_answer).

For better performance with many chunks, use llm_query_batch() which processes prompts in parallel:
```repl
# Split context into chunks and process them in parallel
chunks = [context[i:i+50000] for i in range(0, len(context), 50000)]
prompts = [f"Find any mentions of 'entity' in this chunk: {chunk}" for chunk in chunks]
# This runs all queries in parallel - MUCH faster!
results = llm_query_batch(prompts)
all_mentions = []
for result in results:
    if 'entity' in result.lower():
        all_mentions.append(result)
```

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
"""


def build_system_prompt() -> List[Dict[str, str]]:
    """
    Build the system message for RLM.

    Returns:
        List containing the system message with REPL instructions
    """
    return [
        {
            "role": "system",
            "content": REPL_SYSTEM_PROMPT
        }
    ]


def next_action_prompt(
    query: str,
    iteration: int = 0,
    final_answer: bool = False
) -> Dict[str, str]:
    """
    Generate the user prompt for the next action in the RLM loop.

    Args:
        query: The original user query
        iteration: Current iteration number (0-indexed)
        final_answer: If True, force the model to provide a final answer

    Returns:
        User message dict prompting for next action
    """
    if final_answer:
        return {
            "role": "user",
            "content": (
                "Based on all the information you have gathered, provide a "
                "final answer to the user's query."
            )
        }

    if iteration == 0:
        # First iteration - emphasize looking at context first
        safeguard = (
            "You have not interacted with the REPL environment or seen your "
            "context yet. Your next action should be to look through it, "
            "don't just provide a final answer yet.\n\n"
        )
        base_prompt = (
            f'Think step-by-step on what to do using the REPL environment '
            f'(which contains the context) to answer the original query: '
            f'"{query}".\n\n'
            f'Continue using the REPL environment, which has the `context` '
            f'variable, and querying sub-LLMs by writing to ```repl``` tags '
            f'or shell commands with ```bash/```sh tags, and determine your '
            f'answer. Your next action:'
        )
        return {
            "role": "user",
            "content": safeguard + base_prompt
        }
    else:
        # Subsequent iterations - refer to previous interactions
        base_prompt = (
            f'The history before is your previous interactions with the REPL '
            f'environment. Think step-by-step on what to do using the REPL '
            f'environment (which contains the context) to answer the original '
            f'query: "{query}".\n\n'
            f'Continue using the REPL environment, which has the `context` '
            f'variable, and querying sub-LLMs by writing to ```repl``` tags '
            f'or shell commands with ```bash/```sh tags, and determine your '
            f'answer. Your next action:'
        )
        return {
            "role": "user",
            "content": base_prompt
        }


# Additional helper prompts for specific strategies

PEEKING_EXAMPLE = """
Example: Peeking at the context to understand its structure
```repl
# Check the type and size of context
print(f"Context type: {type(context)}")
print(f"Context size: {len(context) if isinstance(context, (str, list)) else 'N/A'}")

# Peek at the beginning
if isinstance(context, str):
    print(context[:500])
elif isinstance(context, list):
    print(context[:5])
```
"""

GREPPING_EXAMPLE = """
Example: Searching for specific patterns in the context
```repl
import re

# Search for specific keywords
if isinstance(context, str):
    # Find lines containing "magic number"
    matches = re.findall(r'.*magic number.*', context, re.IGNORECASE)
    print(f"Found {len(matches)} matches")
    print(matches[:10])  # Show first 10
```
"""

CHUNKING_EXAMPLE = """
Example: Chunking the context and querying sub-LLMs
```repl
# Chunk the context into manageable pieces
chunk_size = 50000  # ~50k characters per chunk
chunks = []

if isinstance(context, str):
    for i in range(0, len(context), chunk_size):
        chunks.append(context[i:i+chunk_size])

print(f"Created {len(chunks)} chunks")

# Query each chunk
results = []
for i, chunk in enumerate(chunks):
    result = llm_query(f"In this chunk, is there a magic number? If yes, what is it?\\n\\n{chunk}")
    results.append(result)
    print(f"Chunk {i}: {result}")

# Aggregate results
final = llm_query(f"Based on these results, what is the final answer?\\n\\n{results}")
```
"""

"""
ReAct agent baseline with BM25 retrieval.

Implements a simple ReAct (Reasoning + Acting) agent that can
iteratively search and reason over documents.
"""

from typing import Optional, List
from rlm.utils.llm import OpenAIClient
from .retrieval import BM25Retriever


class ReActAgent:
    """
    ReAct agent with BM25 retrieval.

    The agent can:
    - Search: Retrieve relevant documents using BM25
    - Think: Reason about the retrieved information
    - Answer: Provide final answer
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_iterations: int = 5,
        docs_per_search: int = 5,
        api_key: Optional[str] = None
    ):
        """
        Initialize ReAct agent.

        Args:
            model: OpenAI model name
            max_iterations: Maximum reasoning iterations
            docs_per_search: Documents to retrieve per search
            api_key: OpenAI API key
        """
        self.model = model
        self.max_iterations = max_iterations
        self.docs_per_search = docs_per_search
        self.client = OpenAIClient(api_key=api_key, model=model)
        self.retriever = BM25Retriever(documents_per_query=docs_per_search)

    def __call__(self, context: str, query: str) -> str:
        """
        Answer query using ReAct loop with retrieval.

        Args:
            context: Full context (will be indexed for retrieval)
            query: User query

        Returns:
            Final answer
        """
        # Index context for retrieval
        self.retriever.index_context(context, chunk_size=1000)

        # Initialize conversation
        messages = []
        system_prompt = """You are a helpful assistant that answers questions using a search tool.

You have access to the following action:
- Search[query]: Search for relevant information

You should:
1. Think about what information you need
2. Search for that information
3. Think about the results
4. Either search again or provide a final answer

Format your response as:
Thought: [your reasoning]
Action: Search[search query]

OR

Thought: [your reasoning]
Answer: [final answer]"""

        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"Question: {query}"})

        # ReAct loop
        for iteration in range(self.max_iterations):
            # Get agent's response
            try:
                response = self.client.completion(messages=messages)
            except Exception as e:
                return f"ERROR: {str(e)}"

            # Check if agent provided final answer
            if "Answer:" in response:
                # Extract answer
                answer_parts = response.split("Answer:")
                if len(answer_parts) > 1:
                    return answer_parts[1].strip()

            # Check if agent wants to search
            if "Search[" in response:
                # Extract search query
                search_start = response.find("Search[") + 7
                search_end = response.find("]", search_start)

                if search_end == -1:
                    # Malformed search, try to recover
                    search_query = response[search_start:].strip()
                else:
                    search_query = response[search_start:search_end].strip()

                # Perform search
                retrieved_chunks = self.retriever.retrieve_from_context(
                    context,
                    search_query,
                    top_k=self.docs_per_search
                )

                # Format results
                search_results = "\n\n".join([
                    f"Document {i+1}:\n{chunk}"
                    for i, chunk in enumerate(retrieved_chunks)
                ])

                # Add to conversation
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Search Results:\n{search_results}\n\nContinue your reasoning or provide an answer."
                })

            else:
                # Agent didn't search or answer, prompt them
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "Please either Search[query] or provide an Answer:"
                })

        # Max iterations reached, return last response
        return response if response else "ERROR: No answer generated"


class DirectGPTWithBM25:
    """
    Direct GPT model with BM25 pre-retrieval.

    First retrieves top-k documents using BM25, then passes them
    to the LLM with the query.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        top_k_docs: int = 40,
        api_key: Optional[str] = None
    ):
        """
        Initialize DirectGPT with BM25.

        Args:
            model: OpenAI model name
            top_k_docs: Number of documents to retrieve
            api_key: OpenAI API key
        """
        self.model = model
        self.top_k_docs = top_k_docs
        self.client = OpenAIClient(api_key=api_key, model=model)
        self.retriever = BM25Retriever(documents_per_query=top_k_docs)

    def __call__(self, context: str, query: str) -> str:
        """
        Answer query using BM25 retrieval + direct LLM call.

        Args:
            context: Full context
            query: User query

        Returns:
            Model's answer
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve_from_context(
            context,
            query,
            top_k=self.top_k_docs,
            chunk_size=1000
        )

        # Format retrieved context
        retrieved_context = "\n\n".join([
            f"Document {i+1}:\n{chunk}"
            for i, chunk in enumerate(retrieved_chunks)
        ])

        # Create prompt
        prompt = f"""Context (Top {self.top_k_docs} relevant documents):
{retrieved_context}

Question: {query}

Answer:"""

        # Call LLM
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.completion(messages=messages)
            return response
        except Exception as e:
            return f"ERROR: {str(e)}"

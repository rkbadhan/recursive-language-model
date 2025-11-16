"""
BM25-based retrieval baseline.

Simple keyword-based retrieval using BM25 algorithm.
"""

import re
import math
from typing import List, Dict, Tuple
from collections import Counter


class BM25Retriever:
    """
    Simple BM25 retrieval implementation.

    BM25 is a bag-of-words retrieval function that ranks documents
    based on query term frequency.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        documents_per_query: int = 10
    ):
        """
        Initialize BM25 retriever.

        Args:
            k1: Term frequency saturation parameter
            b: Document length normalization parameter
            documents_per_query: Number of documents to retrieve
        """
        self.k1 = k1
        self.b = b
        self.documents_per_query = documents_per_query

        # Will be set during indexing
        self.documents = []
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_freqs = Counter()
        self.idf = {}
        self.N = 0

    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization.

        Args:
            text: Input text

        Returns:
            List of tokens (lowercased words)
        """
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\w+', text.lower())
        return tokens

    def index(self, documents: List[str]) -> None:
        """
        Index documents for BM25 retrieval.

        Args:
            documents: List of document strings
        """
        self.documents = documents
        self.N = len(documents)

        # Tokenize documents and compute lengths
        tokenized_docs = []
        for doc in documents:
            tokens = self.tokenize(doc)
            tokenized_docs.append(tokens)
            self.doc_lengths.append(len(tokens))

        # Compute average document length
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0

        # Compute document frequencies
        self.doc_freqs = Counter()
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

        # Compute IDF scores
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)

    def score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        """
        Compute BM25 score for a document given a query.

        Args:
            query_tokens: Tokenized query
            doc_idx: Document index

        Returns:
            BM25 score
        """
        doc = self.documents[doc_idx]
        doc_tokens = self.tokenize(doc)
        doc_len = self.doc_lengths[doc_idx]

        # Compute term frequencies in document
        term_freqs = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue

            tf = term_freqs.get(term, 0)
            idf = self.idf[term]

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))

            score += idf * (numerator / denominator)

        return score

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        """
        Retrieve top-k documents for a query.

        Args:
            query: Query string
            top_k: Number of documents to retrieve (default: self.documents_per_query)

        Returns:
            List of (doc_idx, score) tuples, sorted by score descending
        """
        if top_k is None:
            top_k = self.documents_per_query

        query_tokens = self.tokenize(query)

        # Score all documents
        scores = []
        for idx in range(self.N):
            score = self.score_document(query_tokens, idx)
            scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Chunk text into smaller segments.

        Args:
            text: Input text
            chunk_size: Approximate chunk size in characters

        Returns:
            List of text chunks
        """
        # Simple chunking by splitting on newlines and grouping
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line)

            if current_size + line_size > chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(line)
            current_size += line_size

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def index_context(self, context: str, chunk_size: int = 500) -> None:
        """
        Index a large context by chunking it.

        Args:
            context: Full context string
            chunk_size: Size of each chunk
        """
        chunks = self.chunk_text(context, chunk_size=chunk_size)
        self.index(chunks)

    def retrieve_from_context(
        self,
        context: str,
        query: str,
        top_k: int = None,
        chunk_size: int = 500
    ) -> List[str]:
        """
        Retrieve relevant chunks from context.

        Args:
            context: Full context string
            query: Query string
            top_k: Number of chunks to retrieve
            chunk_size: Chunk size for indexing

        Returns:
            List of relevant text chunks
        """
        # Index context
        self.index_context(context, chunk_size=chunk_size)

        # Retrieve top-k chunks
        results = self.retrieve(query, top_k=top_k)

        # Return chunk texts
        retrieved_chunks = [self.documents[idx] for idx, _ in results]

        return retrieved_chunks

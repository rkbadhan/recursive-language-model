"""
Data generator for BrowseComp-Plus benchmark.

Generates synthetic multi-document corpora with multi-hop queries
that require information from multiple documents.
"""

import random
from typing import Dict, List, Tuple


# Sample entities for document generation
COMPANIES = ["TechCorp", "DataFlow", "CloudSync", "AIVentures", "SecureNet"]
PEOPLE = ["Alice Johnson", "Bob Smith", "Carol Chen", "David Rodriguez", "Emma Wilson"]
PRODUCTS = ["AutoML Pro", "DataViz Suite", "Cloud Platform", "SecureAuth", "Analytics Engine"]
LOCATIONS = ["San Francisco", "New York", "London", "Tokyo", "Singapore"]
YEARS = list(range(2015, 2025))


def generate_document(
    doc_id: int,
    topic: str,
    include_answer_fragment: bool = False,
    answer_data: Dict[str, any] = None
) -> Dict[str, str]:
    """
    Generate a single document.

    Args:
        doc_id: Document ID
        topic: Document topic category
        include_answer_fragment: Whether to include part of the answer
        answer_data: Data about the answer to embed

    Returns:
        Dictionary with document metadata
    """
    templates = {
        'company_overview': [
            "{company} was founded in {year} by {person}. The company specializes in {product} and operates primarily in {location}.",
            "{company} is a leading technology company established in {year}. Founded by {person}, it has grown to become a major player in the {product} market.",
            "Based in {location}, {company} was created by {person} in {year}. The company focuses on developing {product} solutions."
        ],
        'product_launch': [
            "In {year}, {company} launched {product}, which received positive reviews from industry analysts.",
            "{company} introduced {product} in {year}, marking a significant milestone for the company.",
            "The launch of {product} by {company} in {year} disrupted the market and attracted significant attention."
        ],
        'financial': [
            "{company} reported revenue of ${revenue}M in Q{quarter} {year}, representing {growth}% growth year-over-year.",
            "Financial results for {company} showed ${revenue}M in revenue for Q{quarter} {year}, with {growth}% growth.",
            "In Q{quarter} {year}, {company} achieved ${revenue}M in revenue, up {growth}% from the previous year."
        ],
        'technology': [
            "{product} utilizes advanced {tech} technology to provide {benefit} for users.",
            "The underlying technology of {product} is based on {tech}, enabling {benefit}.",
            "Built on {tech}, {product} offers {benefit} through innovative design."
        ],
        'people': [
            "{person} serves as {role} at {company}, leading the {department} team.",
            "Under the leadership of {person}, {role} of {company}, the {department} has expanded significantly.",
            "{person} joined {company} as {role} and has been instrumental in growing the {department}."
        ],
        'acquisition': [
            "In {year}, {company} acquired {target} for ${amount}M, expanding its {capability} capabilities.",
            "{company} completed the acquisition of {target} in {year}, paying ${amount}M for the company.",
            "The {year} acquisition of {target} by {company} for ${amount}M strengthened its position in {market}."
        ]
    }

    # Generate content based on topic
    if topic not in templates:
        topic = random.choice(list(templates.keys()))

    template = random.choice(templates[topic])

    # Fill in variables
    content = template.format(
        company=random.choice(COMPANIES),
        person=random.choice(PEOPLE),
        product=random.choice(PRODUCTS),
        location=random.choice(LOCATIONS),
        year=random.choice(YEARS),
        revenue=random.randint(10, 500),
        quarter=random.randint(1, 4),
        growth=random.randint(5, 50),
        tech=random.choice(["machine learning", "blockchain", "cloud computing", "artificial intelligence"]),
        benefit=random.choice(["improved efficiency", "cost savings", "better performance", "enhanced security"]),
        role=random.choice(["CEO", "CTO", "VP of Engineering", "Chief Data Officer"]),
        department=random.choice(["engineering", "data science", "product", "operations"]),
        target=random.choice(["SmallTech", "StartupCo", "InnovateLabs", "DataMasters"]),
        amount=random.randint(10, 1000),
        capability=random.choice(["analytics", "security", "infrastructure", "AI"]),
        market=random.choice(["enterprise software", "cloud services", "AI", "cybersecurity"])
    )

    # If this should include an answer fragment, embed it
    if include_answer_fragment and answer_data:
        content = embed_answer_fragment(content, answer_data)

    return {
        'id': doc_id,
        'title': f"Document {doc_id}: {topic.replace('_', ' ').title()}",
        'content': content,
        'topic': topic
    }


def embed_answer_fragment(content: str, answer_data: Dict[str, any]) -> str:
    """
    Embed an answer fragment into document content.

    Args:
        content: Original content
        answer_data: Answer information to embed

    Returns:
        Modified content with answer fragment
    """
    # Add answer hint at the end
    fragment = answer_data.get('fragment', '')
    if fragment:
        content += f" {fragment}"

    return content


def generate_document_corpus(
    num_documents: int = 100,
    num_evidence_docs: int = 3,
    answer_data: Dict[str, any] = None
) -> Tuple[List[Dict[str, str]], List[int]]:
    """
    Generate a corpus of documents.

    Args:
        num_documents: Total number of documents
        num_evidence_docs: Number of documents containing evidence
        answer_data: Answer information to embed in evidence docs

    Returns:
        Tuple of (documents list, evidence doc IDs)
    """
    topics = ['company_overview', 'product_launch', 'financial', 'technology', 'people', 'acquisition']

    documents = []
    evidence_doc_ids = random.sample(range(num_documents), num_evidence_docs)

    for i in range(num_documents):
        topic = random.choice(topics)
        is_evidence = i in evidence_doc_ids

        doc = generate_document(
            doc_id=i,
            topic=topic,
            include_answer_fragment=is_evidence,
            answer_data=answer_data if is_evidence else None
        )

        documents.append(doc)

    return documents, evidence_doc_ids


def generate_multi_hop_query() -> Dict[str, any]:
    """
    Generate a multi-hop query that requires information from multiple documents.

    Returns:
        Dictionary with query, answer, and metadata
    """
    # Select random entities for the query
    company = random.choice(COMPANIES)
    product = random.choice(PRODUCTS)
    person = random.choice(PEOPLE)
    year = random.choice(YEARS)

    queries = [
        {
            'query': f"What product did {company} launch in {year}, and who was the CEO at that time?",
            'answer': f"{product} and {person}",
            'fragments': [
                f"{company} launched {product} in {year}",
                f"{person} serves as CEO at {company}"
            ]
        },
        {
            'query': f"Which company founded by {person} operates in {random.choice(LOCATIONS)} and what is their main product?",
            'answer': f"{company} with {product}",
            'fragments': [
                f"{company} was founded by {person}",
                f"operates in {random.choice(LOCATIONS)}",
                f"main product is {product}"
            ]
        },
        {
            'query': f"What technology does {product} use and which company created it?",
            'answer': f"{random.choice(['machine learning', 'blockchain', 'AI'])} by {company}",
            'fragments': [
                f"{product} utilizes {random.choice(['machine learning', 'blockchain', 'AI'])}",
                f"created by {company}"
            ]
        }
    ]

    query_data = random.choice(queries)

    # Create answer data for embedding
    answer_data = {
        'answer': query_data['answer'],
        'fragment': random.choice(query_data['fragments'])
    }

    return {
        'query': query_data['query'],
        'answer': query_data['answer'],
        'answer_data': answer_data
    }


def format_documents_for_context(documents: List[Dict[str, str]]) -> str:
    """
    Format documents into a single context string.

    Args:
        documents: List of document dictionaries

    Returns:
        Formatted context string
    """
    formatted_docs = []

    for doc in documents:
        formatted = f"=== {doc['title']} ===\n{doc['content']}\n"
        formatted_docs.append(formatted)

    return "\n".join(formatted_docs)


def generate_full_browsecomp_example(
    num_documents: int = 100,
    num_evidence_docs: int = 3
) -> Dict[str, any]:
    """
    Generate a complete BrowseComp-Plus example.

    Args:
        num_documents: Number of documents in corpus
        num_evidence_docs: Number of documents with evidence

    Returns:
        Dictionary with query, context, and answer
    """
    # Generate query
    query_info = generate_multi_hop_query()

    # Generate corpus with embedded answer
    documents, evidence_ids = generate_document_corpus(
        num_documents=num_documents,
        num_evidence_docs=num_evidence_docs,
        answer_data=query_info['answer_data']
    )

    # Format context
    context = format_documents_for_context(documents)

    return {
        'context': context,
        'query': query_info['query'],
        'answer': query_info['answer'],
        'num_documents': num_documents,
        'evidence_doc_ids': evidence_ids,
        'context_length': len(context)
    }

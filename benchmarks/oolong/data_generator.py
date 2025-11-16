"""
Data generator for OOLONG benchmark queries.

Generates synthetic OOLONG-style queries with fine-grained information
that requires semantic understanding and counting.
"""

import random
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


# TREC question categories for classification
TREC_CATEGORIES = [
    "abbreviation",  # Abbreviation questions (What does NASA stand for?)
    "entity",        # Entity questions (Who is X?)
    "description",   # Description/definition questions (What is X?)
    "human",         # Human-related questions (Who invented X?)
    "location",      # Location questions (Where is X?)
    "numeric"        # Numeric questions (How many X?)
]


def generate_sample_questions() -> List[str]:
    """
    Generate a pool of sample TREC-style questions.

    Returns:
        List of question strings
    """
    questions = [
        # Abbreviation
        "What does NASA stand for?",
        "What is the abbreviation for United Nations?",
        "What does HTTP mean?",
        "What is the full form of DNA?",
        "What does AI stand for?",

        # Entity
        "Who is the founder of Microsoft?",
        "What company makes the iPhone?",
        "Who invented the telephone?",
        "What organization runs the Olympics?",
        "Who wrote Harry Potter?",

        # Description
        "What is photosynthesis?",
        "What is a black hole?",
        "What is democracy?",
        "What is machine learning?",
        "What is quantum computing?",

        # Human
        "Who was the first person on the moon?",
        "Who painted the Mona Lisa?",
        "Who discovered penicillin?",
        "Who is the current US President?",
        "Who invented the light bulb?",

        # Location
        "Where is the Eiffel Tower?",
        "Where is Mount Everest?",
        "Where is the Great Wall of China?",
        "Where is the Statue of Liberty?",
        "Where is the Grand Canyon?",

        # Numeric
        "How many planets are in our solar system?",
        "How many continents are there?",
        "How many days in a year?",
        "How many states in the USA?",
        "How many wheels on a bicycle?",
    ]

    # Extend with variations
    extended = questions.copy()
    for _ in range(10):
        extended.extend(questions)

    return extended


def classify_question(question: str) -> str:
    """
    Classify a question into TREC categories.

    This is a simple heuristic classifier based on question keywords.

    Args:
        question: Question string

    Returns:
        TREC category label
    """
    question_lower = question.lower()

    if any(word in question_lower for word in ["stand for", "abbreviation", "acronym", "mean"]):
        return "abbreviation"
    elif question_lower.startswith("who") or "person" in question_lower:
        if any(word in question_lower for word in ["invent", "discover", "paint", "write", "create"]):
            return "human"
        else:
            return "entity"
    elif question_lower.startswith("what is") or question_lower.startswith("what are"):
        return "description"
    elif question_lower.startswith("where"):
        return "location"
    elif question_lower.startswith("how many") or "number" in question_lower:
        return "numeric"
    else:
        # Random fallback
        return random.choice(TREC_CATEGORIES)


def generate_oolong_entries(
    num_entries: int = 5000,
    num_users: int = 1000,
    start_user_id: int = 10000
) -> Tuple[List[str], Dict[int, str]]:
    """
    Generate OOLONG-style entries with metadata.

    Each entry has format:
    Date: [date] || User: [user_id] || Instance: [question]

    Args:
        num_entries: Number of entries to generate
        num_users: Number of unique users
        start_user_id: Starting user ID

    Returns:
        Tuple of (entries list, user_id -> category mapping)
    """
    questions = generate_sample_questions()
    user_ids = list(range(start_user_id, start_user_id + num_users))

    entries = []
    user_categories = {}

    # Generate random dates
    start_date = datetime(2022, 1, 1)

    for i in range(num_entries):
        # Random date in 2022-2024
        days_offset = random.randint(0, 1095)  # ~3 years
        date = start_date + timedelta(days=days_offset)
        date_str = date.strftime("%b %d, %Y")

        # Random user
        user_id = random.choice(user_ids)

        # Random question
        question = random.choice(questions)

        # Classify question
        category = classify_question(question)

        # Track category for this user/entry
        user_categories[(user_id, i)] = category

        # Format entry
        entry = f"Date: {date_str} || User: {user_id} || Instance: {question}"
        entries.append(entry)

    return entries, user_categories


def generate_oolong_query(
    entries: List[str],
    user_categories: Dict[Tuple[int, int], str],
    target_label: str = "entity",
    num_target_users: int = 20
) -> Tuple[str, int, List[int]]:
    """
    Generate an OOLONG-style query for the given entries.

    Args:
        entries: List of data entries
        user_categories: Mapping of (user_id, entry_idx) to category
        target_label: Category label to count
        num_target_users: Number of users to include in query

    Returns:
        Tuple of (query_text, expected_count, target_user_ids)
    """
    # Extract all user IDs
    all_user_ids = set()
    for (user_id, _), _ in user_categories.items():
        all_user_ids.add(user_id)

    # Sample target users
    target_user_ids = sorted(random.sample(list(all_user_ids), num_target_users))

    # Count entries matching criteria
    count = 0
    for (user_id, entry_idx), category in user_categories.items():
        if user_id in target_user_ids and category == target_label:
            count += 1

    # Generate query text
    user_ids_str = ", ".join(map(str, target_user_ids))
    query = (
        f"For the following question, only consider the subset of instances that are "
        f"associated with user IDs {user_ids_str}. "
        f"Among instances associated with these users, how many data points should be "
        f"classified as label '{target_label}'? "
        f"Give your final answer in the form 'Answer: number'."
    )

    return query, count, target_user_ids


def create_oolong_context(entries: List[str]) -> str:
    """
    Create the full context string from entries.

    Args:
        entries: List of entry strings

    Returns:
        Full context as a single string
    """
    return "\n".join(entries)


def generate_full_oolong_example(
    num_entries: int = 5000,
    target_label: str = "entity",
    num_target_users: int = 20
) -> Dict[str, any]:
    """
    Generate a complete OOLONG example with context and query.

    Args:
        num_entries: Number of data entries
        target_label: Category label to count
        num_target_users: Number of users in query

    Returns:
        Dictionary with 'context', 'query', 'answer', 'target_users'
    """
    # Generate entries
    entries, user_categories = generate_oolong_entries(num_entries=num_entries)

    # Generate query
    query, count, target_user_ids = generate_oolong_query(
        entries,
        user_categories,
        target_label=target_label,
        num_target_users=num_target_users
    )

    # Create context
    context = create_oolong_context(entries)

    return {
        'context': context,
        'query': query,
        'answer': count,
        'target_users': target_user_ids,
        'target_label': target_label,
        'num_entries': num_entries,
        'context_length': len(context)
    }

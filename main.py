"""
Example demonstrations of Recursive Language Models (RLM).

This module showcases various use cases of RLMs, including:
- Needle-in-haystack tasks
- Multi-document reasoning
- Counting and aggregation
- Semantic search over large corpora
"""

import random
import os
from typing import List
from rlm.rlm_repl import RLM_REPL


def example_1_needle_in_haystack():
    """
    Example 1: Needle-in-Haystack

    Generate a massive context (1M lines) and hide a magic number somewhere.
    The RLM must find it by exploring the context programmatically.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Needle-in-Haystack (1M lines)")
    print("="*80)

    # Generate the answer
    answer = str(random.randint(1000000, 9999999))
    print(f"Hidden answer: {answer}")

    # Generate massive context
    print("Generating 1M lines of random text...")
    random_words = ["blah", "random", "text", "data", "content", "information", "sample"]
    lines = []

    for _ in range(1_000_000):
        num_words = random.randint(3, 8)
        line_words = [random.choice(random_words) for _ in range(num_words)]
        lines.append(" ".join(line_words))

    # Insert the magic number at a random position
    magic_position = random.randint(400000, 600000)
    lines[magic_position] = f"The magic number is {answer}"
    print(f"Magic number inserted at line {magic_position}")

    context = "\n".join(lines)
    print(f"Context size: {len(context):,} characters")

    # Create RLM and query
    rlm = RLM_REPL(
        model="gpt-4o-mini",
        recursive_model="gpt-4o-mini",
        enable_logging=True,
        max_iterations=10,
        track_costs=True
    )

    query = "Find the magic number in the context. What is it?"

    print(f"\nQuery: {query}")
    print("\nRunning RLM...\n")

    result = rlm.completion(context=context, query=query)

    print("\n" + "="*80)
    print(f"Result: {result}")
    print(f"Expected: {answer}")
    print(f"Correct: {answer in result}")

    # Show cost summary
    if rlm.track_costs:
        cost_summary = rlm.cost_summary()
        print(f"\nCost Summary:")
        print(f"  Total calls: {cost_summary['total_calls']}")
        print(f"  Total tokens: {cost_summary['total_tokens']:,}")
        print(f"  Estimated cost: ${cost_summary['estimated_cost_usd']:.4f}")

    print("="*80)


def example_2_multi_document():
    """
    Example 2: Multi-Document Reasoning

    Create multiple documents and ask a question that requires
    information from multiple sources.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Multi-Document Reasoning")
    print("="*80)

    # Create sample documents
    documents = [
        {
            "title": "Company Overview",
            "content": "TechCorp was founded in 2010 by Alice Johnson and Bob Smith. "
                      "The company specializes in AI and machine learning solutions."
        },
        {
            "title": "Financial Report Q1 2024",
            "content": "TechCorp reported revenue of $50M in Q1 2024, up 25% from Q1 2023. "
                      "The growth was driven primarily by the AI division."
        },
        {
            "title": "Product Launches",
            "content": "In March 2024, TechCorp launched AutoML Pro, an automated machine "
                      "learning platform. The product received positive reviews."
        },
        {
            "title": "Team Updates",
            "content": "Alice Johnson, co-founder and CEO, announced plans to expand the "
                      "engineering team by 50% in 2024. Bob Smith serves as CTO."
        },
        {
            "title": "Market Analysis",
            "content": "The AI market is expected to grow at 35% CAGR through 2028. "
                      "TechCorp is well-positioned to capture market share."
        },
    ]

    # Repeat documents to make context larger
    all_documents = []
    for i in range(20):
        for doc in documents:
            all_documents.append(f"Document {i*len(documents) + documents.index(doc)}: {doc['title']}\n{doc['content']}\n")

    context = "\n".join(all_documents)
    print(f"Context: {len(all_documents)} documents, {len(context):,} characters")

    query = (
        "Who are the founders of TechCorp, what product did they launch in 2024, "
        "and what was their Q1 2024 revenue?"
    )

    rlm = RLM_REPL(
        model="gpt-4o-mini",
        recursive_model="gpt-4o-mini",
        enable_logging=True,
        max_iterations=8
    )

    print(f"\nQuery: {query}")
    print("\nRunning RLM...\n")

    result = rlm.completion(context=context, query=query)

    print("\n" + "="*80)
    print(f"Result: {result}")
    print("="*80)


def example_3_counting_aggregation():
    """
    Example 3: Counting and Aggregation (OOLONG-style)

    Create a list of entries with metadata and ask the model to count
    entries matching specific criteria.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Counting and Aggregation")
    print("="*80)

    # Generate sample data
    categories = ["entity", "description", "abbreviation", "number", "location"]
    user_ids = list(range(1000, 2000))

    entries = []
    target_user_ids = [1234, 1456, 1789, 1999, 1100]
    entity_count = 0

    for i in range(5000):
        user_id = random.choice(user_ids)
        category = random.choice(categories)

        # Track entities for target users
        if user_id in target_user_ids and category == "entity":
            entity_count += 1

        entries.append(
            f"ID: {i} || User: {user_id} || Category: {category} || "
            f"Question: What is example {i}?"
        )

    context = "\n".join(entries)
    print(f"Generated {len(entries)} entries")
    print(f"Target user IDs: {target_user_ids}")
    print(f"Expected count of 'entity' for target users: {entity_count}")

    query = (
        f"Only consider entries associated with user IDs {target_user_ids}. "
        f"Among these entries, how many have category 'entity'? "
        f"Give your final answer as a number."
    )

    rlm = RLM_REPL(
        model="gpt-4o-mini",
        recursive_model="gpt-4o-mini",
        enable_logging=True,
        max_iterations=10
    )

    print(f"\nQuery: {query}")
    print("\nRunning RLM...\n")

    result = rlm.completion(context=context, query=query)

    print("\n" + "="*80)
    print(f"Result: {result}")
    print(f"Expected: {entity_count}")
    print("="*80)


def example_4_simple_test():
    """
    Example 4: Simple Test

    A minimal example to test basic functionality.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Simple Test")
    print("="*80)

    context = """
    Alice has 5 apples.
    Bob has 3 oranges.
    Charlie has 7 bananas.
    Diana has 2 pears.
    """

    query = "How many total fruits do all people have?"

    rlm = RLM_REPL(
        model="gpt-4o-mini",
        recursive_model="gpt-4o-mini",
        enable_logging=True,
        max_iterations=5
    )

    print(f"Context: {context}")
    print(f"\nQuery: {query}")
    print("\nRunning RLM...\n")

    result = rlm.completion(context=context, query=query)

    print("\n" + "="*80)
    print(f"Result: {result}")
    print(f"Expected: 17 fruits")
    print("="*80)


def example_5_log_analysis():
    """
    Example 5: System Log Analysis and Correlation

    Demonstrate log parsing and correlation capabilities.
    This example runs WITHOUT requiring an API key - it shows
    the parsing and correlation features standalone.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: System Log Analysis (No API Key Required)")
    print("="*80)
    print()
    print("This example demonstrates:")
    print("  - Parsing different log formats (jstack, strace, GC, syslog)")
    print("  - Correlating events across multiple logs")
    print("  - Detecting known issue patterns")
    print("  - Building timelines and root cause analysis")
    print()
    print("Note: This runs the parser demo which doesn't need OpenAI API.")
    print("For full AI-powered analysis, run: python demo_log_analysis.py")
    print()

    # Import log parsing modules
    from rlm.log_parsers import parse_log, detect_log_format
    from rlm.log_correlator import correlate_logs, detect_all_patterns

    # Sample logs (simplified for quick demo)
    jstack_log = '''
"Worker-1" #12 prio=5 os_prio=0 tid=0x00007f8a4c000800 nid=0x1a2b waiting on condition
   java.lang.Thread.State: BLOCKED (on object monitor)
    at com.example.Service.process(Service.java:45)
    - waiting to lock <0x00000000e1234560> (a java.lang.Object)

Found one Java-level deadlock:
"Worker-1": waiting to lock object 0x00000000e1234560
'''

    gc_log = '''
[2024-11-15T14:30:15.456+0000][gc] GC(101) Pause Young (Normal) 55M->15M(100M) 5234.567ms
[2024-11-15T14:31:00.000+0000][gc] GC(103) Pause Full (Ergonomics) 90M->30M(100M) 8000.123ms
'''

    syslog_log = '''
Nov 15 14:30:15 hostname application[12346]: ERROR: Database connection failed
Nov 15 14:30:16 hostname application[12346]: WARNING: Retrying connection
'''

    print("Step 1: Parsing logs...")
    print("-" * 40)

    # Parse each log
    jstack_parsed = parse_log(jstack_log)
    print(f"✓ jstack: format={jstack_parsed['format']}, deadlock={jstack_parsed.get('has_deadlock', False)}")

    gc_parsed = parse_log(gc_log)
    print(f"✓ GC: format={gc_parsed['format']}, collections={gc_parsed.get('total_collections', 0)}")

    syslog_parsed = parse_log(syslog_log)
    print(f"✓ syslog: format={syslog_parsed['format']}, errors={syslog_parsed.get('error_count', 0)}")

    print()
    print("Step 2: Correlating logs...")
    print("-" * 40)

    # Correlate
    parsed_logs = {
        'jstack': jstack_parsed,
        'gc': gc_parsed,
        'syslog': syslog_parsed
    }

    timeline = correlate_logs(parsed_logs)
    print(f"✓ Timeline created with {len(timeline.events)} events from {len(timeline.by_source)} sources")

    print()
    print("Step 3: Detecting patterns...")
    print("-" * 40)

    patterns = detect_all_patterns(timeline)
    if patterns:
        for i, p in enumerate(patterns, 1):
            print(f"{i}. [{p['severity']}] {p['pattern'].upper()}: {p['description']}")
    else:
        print("No critical patterns detected in this sample.")

    print()
    print("="*80)
    print("Demo complete! For a comprehensive example with real logs, run:")
    print("  python demo_log_analysis.py")
    print("="*80)


def main():
    """Run examples based on user selection."""
    print("\n" + "="*80)
    print("RECURSIVE LANGUAGE MODEL (RLM) - EXAMPLES")
    print("="*80)
    print("\nAvailable examples:")
    print("  1. Needle-in-Haystack (1M lines) - Intensive, slow")
    print("  2. Multi-Document Reasoning - Moderate")
    print("  3. Counting and Aggregation - Moderate")
    print("  4. Simple Test - Quick")
    print("  5. System Log Analysis - Quick (parser demo, no API needed)")
    print("  6. Run all examples (WARNING: Expensive!)")

    choice = input("\nEnter your choice (1-6): ").strip()

    # Check for API key for examples that need it (not example 5)
    if choice != "5" and not os.getenv("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it in a .env file or export it:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("\nNote: Example 5 (Log Analysis) doesn't require an API key.")
        return

    if choice == "1":
        example_1_needle_in_haystack()
    elif choice == "2":
        example_2_multi_document()
    elif choice == "3":
        example_3_counting_aggregation()
    elif choice == "4":
        example_4_simple_test()
    elif choice == "5":
        example_5_log_analysis()
    elif choice == "6":
        confirm = input(
            "Running all examples will be expensive. Continue? (yes/no): "
        ).strip().lower()
        if confirm == "yes":
            example_4_simple_test()
            example_2_multi_document()
            example_3_counting_aggregation()
            example_5_log_analysis()  # Log analysis is cheap
            # Skip example 1 by default as it's very intensive
            print("\nSkipping Example 1 (Needle-in-Haystack) - too intensive")
            print("Run it separately if needed.")
        else:
            print("Cancelled.")
    else:
        print("Invalid choice. Running simple test...")
        example_4_simple_test()

    print("\n" + "="*80)
    print("DONE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

"""
Test script to verify regex fixes for code blocks and FINAL() patterns.
"""

from rlm.utils.utils import find_code_blocks, find_final_answer


def test_code_blocks():
    """Test code block extraction with various edge cases."""
    print("Testing code block extraction...")

    # Test 1: Standard case
    text1 = """```repl
x = 5
print(x)
```"""
    result1 = find_code_blocks(text1)
    assert result1 == ["x = 5\nprint(x)"], f"Failed test 1: {result1}"
    print("✓ Test 1: Standard code block")

    # Test 2: No trailing newline before closing backticks
    text2 = """```repl
x = 5```"""
    result2 = find_code_blocks(text2)
    assert result2 == ["x = 5"], f"Failed test 2: {result2}"
    print("✓ Test 2: No trailing newline")

    # Test 3: Multiple trailing newlines
    text3 = """```repl
x = 5


```"""
    result3 = find_code_blocks(text3)
    assert result3 == ["x = 5"], f"Failed test 3: {result3}"
    print("✓ Test 3: Multiple trailing newlines")

    # Test 4: Multiple code blocks
    text4 = """```repl
x = 1
```
Some text
```repl
y = 2
```"""
    result4 = find_code_blocks(text4)
    assert result4 == ["x = 1", "y = 2"], f"Failed test 4: {result4}"
    print("✓ Test 4: Multiple code blocks")

    # Test 5: Empty code block
    text5 = """```repl
```"""
    result5 = find_code_blocks(text5)
    assert result5 == [""], f"Failed test 5: {result5}"
    print("✓ Test 5: Empty code block")

    print("\n✅ All code block tests passed!\n")


def test_final_patterns():
    """Test FINAL() pattern extraction with nested parentheses."""
    print("Testing FINAL() pattern extraction...")

    # Test 1: Simple FINAL
    text1 = "FINAL(result)"
    result1 = find_final_answer(text1)
    assert result1 == ('FINAL', 'result'), f"Failed test 1: {result1}"
    print("✓ Test 1: Simple FINAL")

    # Test 2: FINAL with nested parentheses
    text2 = "FINAL(calculate(5 + 3))"
    result2 = find_final_answer(text2)
    assert result2 == ('FINAL', 'calculate(5 + 3)'), f"Failed test 2: {result2}"
    print("✓ Test 2: FINAL with nested parentheses")

    # Test 3: FINAL with multiple nested levels
    text3 = "FINAL(func(nested(deep(value))))"
    result3 = find_final_answer(text3)
    assert result3 == ('FINAL', 'func(nested(deep(value)))'), f"Failed test 3: {result3}"
    print("✓ Test 3: FINAL with multiple nesting levels")

    # Test 4: FINAL_VAR simple
    text4 = "FINAL_VAR(result_var)"
    result4 = find_final_answer(text4)
    assert result4 == ('FINAL_VAR', 'result_var'), f"Failed test 4: {result4}"
    print("✓ Test 4: Simple FINAL_VAR")

    # Test 5: FINAL_VAR with nested parentheses
    text5 = "FINAL_VAR(get_result(x))"
    result5 = find_final_answer(text5)
    assert result5 == ('FINAL_VAR', 'get_result(x)'), f"Failed test 5: {result5}"
    print("✓ Test 5: FINAL_VAR with nested parentheses")

    # Test 6: FINAL_VAR takes precedence over FINAL
    text6 = "FINAL_VAR(var1) and FINAL(var2)"
    result6 = find_final_answer(text6)
    assert result6 == ('FINAL_VAR', 'var1'), f"Failed test 6: {result6}"
    print("✓ Test 6: FINAL_VAR precedence")

    # Test 7: FINAL with quoted string containing parens
    text7 = 'FINAL("result with (parens) inside")'
    result7 = find_final_answer(text7)
    assert result7 == ('FINAL', '"result with (parens) inside"'), f"Failed test 7: {result7}"
    print("✓ Test 7: FINAL with quoted string")

    # Test 8: No FINAL pattern
    text8 = "Just some regular text"
    result8 = find_final_answer(text8)
    assert result8 is None, f"Failed test 8: {result8}"
    print("✓ Test 8: No FINAL pattern")

    print("\n✅ All FINAL() pattern tests passed!\n")


if __name__ == "__main__":
    test_code_blocks()
    test_final_patterns()
    print("=" * 50)
    print("🎉 All tests passed successfully!")
    print("=" * 50)

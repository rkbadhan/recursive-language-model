"""
Simple test script to verify shell command support without dependencies.
"""

import re
import subprocess
import time

def test_code_block_detection():
    """Test detection of bash/sh code blocks."""
    print("Testing code block detection...")

    def find_code_blocks(text):
        """Extract code blocks from text."""
        pattern = r'```(repl|bash|sh)\s*\n(.*?)\n```'
        matches = re.finditer(pattern, text, re.DOTALL)
        results = []
        for match in matches:
            language = match.group(1).strip()
            code_content = match.group(2).strip()
            results.append((language, code_content))
        return results if results else None

    # Test detecting repl blocks
    text1 = "Some text\n```repl\nprint('hello')\n```\nMore text"
    blocks = find_code_blocks(text1)
    print(f"1. Detected blocks from repl: {blocks}")
    assert blocks is not None, "Should detect repl block"
    assert len(blocks) == 1, "Should detect 1 block"
    assert blocks[0][0] == 'repl', "Language should be repl"
    assert "print('hello')" in blocks[0][1], "Should contain code"

    # Test detecting bash blocks
    text2 = "Some text\n```bash\necho 'test'\n```\nMore text"
    blocks = find_code_blocks(text2)
    print(f"2. Detected blocks from bash: {blocks}")
    assert blocks is not None, "Should detect bash block"
    assert len(blocks) == 1, "Should detect 1 block"
    assert blocks[0][0] == 'bash', "Language should be bash"
    assert "echo 'test'" in blocks[0][1], "Should contain command"

    # Test detecting sh blocks
    text3 = "Some text\n```sh\nls -la\n```\nMore text"
    blocks = find_code_blocks(text3)
    print(f"3. Detected blocks from sh: {blocks}")
    assert blocks is not None, "Should detect sh block"
    assert len(blocks) == 1, "Should detect 1 block"
    assert blocks[0][0] == 'sh', "Language should be sh"
    assert "ls -la" in blocks[0][1], "Should contain command"

    # Test detecting mixed blocks
    text4 = """```repl
x = 5
print(x)
```

```bash
echo "hello"
```"""
    blocks = find_code_blocks(text4)
    print(f"4. Detected blocks from mixed: {blocks}")
    assert blocks is not None, "Should detect mixed blocks"
    assert len(blocks) == 2, "Should detect 2 blocks"
    assert blocks[0][0] == 'repl', "First should be repl"
    assert blocks[1][0] == 'bash', "Second should be bash"

    print("✓ All code block detection tests passed!")


def test_shell_execution():
    """Test basic shell command execution."""
    print("\nTesting shell command execution...")

    # Test simple shell command
    print("\n1. Testing simple echo command:")
    start_time = time.time()
    result = subprocess.run(
        "echo 'Hello from shell!'",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        executable='/bin/bash'
    )
    execution_time = time.time() - start_time
    print(f"   stdout: {result.stdout.strip()}")
    print(f"   stderr: {result.stderr}")
    print(f"   returncode: {result.returncode}")
    print(f"   execution_time: {execution_time:.4f}s")
    assert result.returncode == 0, "Echo command should succeed"
    assert "Hello from shell!" in result.stdout, "Output should contain echo message"

    # Test ls command
    print("\n2. Testing ls command:")
    result = subprocess.run(
        "ls -la /tmp",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        executable='/bin/bash'
    )
    print(f"   stdout (first 100 chars): {result.stdout[:100]}...")
    print(f"   returncode: {result.returncode}")
    assert result.returncode == 0, "ls command should succeed"

    # Test command that creates a file
    print("\n3. Testing file creation:")
    result = subprocess.run(
        "echo 'test content' > /tmp/test_rlm.txt && cat /tmp/test_rlm.txt",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        executable='/bin/bash'
    )
    print(f"   stdout: {result.stdout.strip()}")
    print(f"   returncode: {result.returncode}")
    assert result.returncode == 0, "File creation should succeed"
    assert "test content" in result.stdout, "File content should be visible"

    # Test failing command
    print("\n4. Testing failing command:")
    result = subprocess.run(
        "nonexistent_command_xyz",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        executable='/bin/bash'
    )
    print(f"   stdout: {result.stdout}")
    print(f"   stderr (first 100 chars): {result.stderr[:100]}...")
    print(f"   returncode: {result.returncode}")
    assert result.returncode != 0, "Invalid command should fail"

    print("\n✓ All shell execution tests passed!")


if __name__ == "__main__":
    test_code_block_detection()
    test_shell_execution()
    print("\n🎉 All tests passed successfully!")

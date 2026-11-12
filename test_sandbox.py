"""PrivAI - sandbox test. Tries to escape the workspace folder WITHOUT the AI.
Run:  python test_sandbox.py
"""
from pathlib import Path

from tools import read_file, write_file, WORKSPACE

real_file = str(Path("app.py").resolve())   # absolute path to a file outside workspace

attacks = [
    ("Parent folder (../)",        lambda: read_file("../app.py")),
    ("Deep parent (../../)",       lambda: read_file("../../app.py")),
    ("Windows style (..\\)",       lambda: read_file("..\\app.py")),
    ("Absolute path",              lambda: read_file(real_file)),
    ("Write outside workspace",    lambda: write_file("../hacked.txt", "pwned")),
    ("Sneaky path (a/../../x)",    lambda: read_file("sub/../../app.py")),
]

passed = 0
print("=== ATTACK TESTS (all should be BLOCKED) ===")
for name, attack in attacks:
    result = attack()
    blocked = "Access denied" in result
    passed += blocked
    print(f"{'PASS' if blocked else 'FAIL'} | {name:26} -> {result[:60]!r}")

print("\n=== NORMAL USE (should WORK) ===")
print(write_file("test_ok.txt", "hello"))
ok = read_file("test_ok.txt") == "hello"
print(f"{'PASS' if ok else 'FAIL'} | normal read/write inside workspace")
(WORKSPACE / "test_ok.txt").unlink(missing_ok=True)
Path("hacked.txt").unlink(missing_ok=True)

print(f"\nResult: {passed}/{len(attacks)} attacks blocked, normal use works: {ok}")
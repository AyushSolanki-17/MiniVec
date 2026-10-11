# Contributing to Binturong

Thanks for your interest in Binturong. The project is **pre-alpha**: the C++ HNSW
engine is an experimental codebase used to validate correctness and performance
ideas, and its APIs, file layout and repository structure may change
substantially. Please open an issue to discuss larger changes before writing code.

## Building and testing

You need a C++17 compiler (GCC ≥ 11 or Clang ≥ 14), CMake ≥ 3.18 and Python ≥ 3.9.

```bash
# C++ library, tests and Python extension
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(command -v python3)"
cmake --build build -j

# C++ tests (the long-running stress test is excluded by default)
ctest --test-dir build --output-on-failure -j4 -LE stress

# Python tests against the in-tree build
PYTHONPATH=build:$PYTHONPATH python3 -m pytest -q

# Lint
ruff check .
```

The Python extension is built for one specific interpreter. If `import binturong_cpp`
fails, reconfigure with `-DPython_EXECUTABLE` pointing at the Python you test with.

For sanitizer builds, configure with `-DBINTURONG_SANITIZE=address` (ASAN + UBSAN) or
`-DBINTURONG_SANITIZE=thread` (TSAN) in a separate build directory.

## Pull requests

- Keep each pull request to one logical change.
- Every behavior change needs a test; every bug fix needs a test that fails without
  the fix.
- Don't weaken an existing test or threshold to make a change pass. If a threshold
  is wrong, explain why in the PR.
- Performance or recall claims in docs must be measured: state the dataset,
  parameters, hardware and commit.
- Don't add third-party dependencies without discussing them first.
- Follow `.editorconfig` and match the style of the surrounding code.
- Pull request CI runs lint, a Linux build with the quick tests, and the Python
  tests. Sanitizer, macOS and stress runs happen in a separate scheduled workflow.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `perf:`, `test:`, `ci:`, `docs:`, `chore:`, `refactor:`, `bench:`.
Describe what changed and why, for example `fix: keep long-range edges when pruning`.

## Developer Certificate of Origin

Contributions are accepted under the [Developer Certificate of Origin](https://developercertificate.org/)
(DCO). Every commit must carry a `Signed-off-by` line with your real name and email,
certifying that you wrote the change or otherwise have the right to submit it:

```
Signed-off-by: Jane Doe <jane@example.com>
```

`git commit -s` adds it automatically.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE), the same license that covers the project.

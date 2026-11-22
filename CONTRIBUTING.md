# Contributing to AB Wildcard

Thanks for helping. Three steps:

1. **Open an issue** for anything bigger than a typo, so we can agree on the syntax or behaviour before you write code.
2. **Fork, branch, change.** Keep one topic per pull request. New syntax needs a test in `tests/test_processing.py` and a line in the README's Syntax Reference.
3. **Run the tests**, then open the pull request:

   ```bash
   pip install -r requirements-dev.txt
   pytest
   node tests/syntax.test.mjs
   ```

Found a bug? An issue with the prompt, the seed and what you expected is enough. A workflow JSON helps even more.

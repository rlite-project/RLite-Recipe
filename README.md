# RLite Recipe

🚀 A curated collection of training recipes for [RLite](https://github.com/rlite-project/RLite) 📜✨, designed to provide comprehensive guidance on implementing reinforcement learning (RL) techniques for large language models (LLMs) and vision-language models (VLMs). This repository offers modular configurations ⚙️, training workflows 📊, and best practices 🌟 to help users reproduce state-of-the-art RL-driven results 🏆 in alignment, fine-tuning, and optimization tasks.

🌱 While not exhaustive, the current recipes focus on foundational and emerging methodologies, serving as a starting point for adapting RLite to custom projects. Contributions are welcome 🙌 to expand coverage of existing research and benchmarks.

🚧 Under active development — we aim to expand support collaboratively while maintaining reproducibility 🔍✅ and clarity ✨📖. New recipes will be added progressively as the ecosystem evolves!

*"Cooking up RL innovations, one recipe at a time!"* 👩🍳🔥

## Contributing

<details>
<summary>Developer's guide.</summary>

We use `pre-commit` and `git cz` to sanitize the commits. You can run `pre-commit` before `git cz` to avoid repeatedly input the commit messages.

```bash
pip install pre-commit
# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
# Install this emoji-style tool
sudo npm install -g git-cz --no-audit --verbose --registry=https://registry.npmmirror.com

# Install rlite and development dependencies
pip install -e ".[dev]"
```

##### Code Style

- Single line code length is 99 characters, comments and documents are 79 characters.
- Write unit tests for atomic capabilities to ensure that `pytest` does not throw an error.

Run `pre-commit` to automatically lint the code:

```bash
pre-commit run --all-files
```

</details>

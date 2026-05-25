# Contributing to `metriclab`

We want to make contributing to this project as easy and transparent as possible.

## Overview

We welcome contributions from anyone, even if you are new to open source.

- If you are planning to contribute back bug-fixes, please do so without any further discussion.
- If you plan to contribute new features, utility functions, or extensions to the core, please first
  open an issue and discuss the feature with us.

Once you implement and test your feature or bug-fix, please submit a Pull Request.

## Development Setup

### Prerequisites

- Python 3.10 or later
- Git
- `uv` for dependency management (recommended) or `pip`

### Setting up your development environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```shell
   git clone git@github.com:YOUR_USERNAME/metriclab.git
   cd metriclab
   ```

3. Set up the development environment:
   ```shell
   make setup-venv
   ```

   This will create a virtual environment and install all dependencies.

4. Activate the virtual environment:
   ```shell
   source .venv/bin/activate
   ```

5. Install pre-commit hooks:
   ```shell
   pre-commit install
   ```

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. Make your changes with clear, descriptive commit messages.
3. If you've added code that should be tested, add tests.
4. If you've changed APIs, update the documentation.
5. Ensure the test suite passes:
   ```shell
   make unit-test-cov
   ```
6. Make sure your code lints:
   ```shell
   inv check-lint
   ```
7. Format your code:
   ```shell
   inv check-format
   ```
8. Run the pre-commit hooks:
   ```shell
   pre-commit run --all-files
   ```
9. Push to your fork and submit a pull request to the `main` branch.

### Pull Request Guidelines

- Keep pull requests focused on a single issue or feature
- Write clear, descriptive commit messages
- Include tests for new functionality
- Update documentation as needed
- Ensure all CI checks pass
- Reference any related issues in the PR description

## Code Style

This project follows these style guidelines:

- **Code formatting**: [Black](https://github.com/psf/black) with 100 character line length
- **Linting**: [Ruff](https://github.com/astral-sh/ruff)
- **Docstrings
  **: [Google style](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings)
- **Type hints**: Required for all public APIs

Pre-commit hooks will automatically check formatting and run linters.

## Testing

### Running Tests

Run all tests:

```shell
inv unit-test --cov
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names that explain what is being tested
- Follow the existing test structure and patterns
- Aim for high code coverage
- Test edge cases and error conditions

## Documentation

### Building Documentation

Build the documentation locally:

```shell
cd docs
mike serve
```

Then open http://localhost:8000 in your browser.

### Documentation Guidelines

- Update docstrings for any changed functions/classes
- Add examples to docstrings when helpful
- Update user guide pages for API changes
- Keep documentation clear and concise

## Issues

We use GitHub issues to track public bugs or feature requests.

### Reporting Bugs

When reporting bugs, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Python version and metriclab version
- Minimal code example that demonstrates the issue
- Error messages and stack traces (if applicable)

### Feature Requests

For feature requests, please include:

- A clear, concise description of the feature
- Motivation: why is this feature needed?
- Use case examples
- Proposed API or implementation (if applicable)
- Any alternatives you've considered

## Code Review Process

1. A maintainer will review your PR
2. You may be asked to make changes
3. Once approved, a maintainer will merge your PR

Please be patient - maintainers review PRs as time permits.

## Community Guidelines

- Be respectful and constructive
- Welcome newcomers and help them get started
- Focus on the code, not the person
- Assume good intentions

We follow the
[Code of Conduct](https://github.com/durandtibo/metriclab/blob/main/CODE_OF_CONDUCT.md).

## Release Process

Releases are handled by maintainers:

1. Update version in `pyproject.toml`
2. Create a git tag
3. Publish to PyPI
4. Update documentation

## Getting Help

- Browse [existing issues](https://github.com/durandtibo/metriclab/issues)
- Open a [discussion](https://github.com/durandtibo/metriclab/discussions)

## License

By contributing to `metriclab`, you agree that your contributions will be licensed under the LICENSE
file in the root directory of this source tree.

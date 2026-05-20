# GitHub Actions Workflows

This document provides an overview of the GitHub Actions workflows and the structure of the
`.github` folder in the `metriclab` repository.

## Overview

The repository uses GitHub Actions for continuous integration, testing, documentation deployment,
and package publishing. The workflows are designed to be modular and reusable, with a custom
composite action for setting up the development environment.

## `.github` Folder Structure

```
.github/
├── actions/                   # Custom composite actions
│   └── setup-env/             # Reusable action for environment setup
│       └── action.yaml
├── workflows/                 # GitHub Actions workflow definitions
│   ├── build.yaml             # Build and package tests
│   ├── ci.yaml                # Main CI workflow
│   ├── coverage.yaml          # Code coverage reporting
│   ├── test.yaml              # Unit and integration tests
│   ├── docs.yaml              # Documentation deployment
│   ├── pypi.yaml              # PyPI package publishing
│   └── ...                    # Additional workflows
├── CONTRIBUTING.md            # Contribution guidelines
├── ISSUE_TEMPLATE/            # Issue templates
│   ├── bug-report.yml
│   └── feature-request.yml
├── PULL_REQUEST_TEMPLATE.md   # Pull request template
└── dependabot.yml             # Dependabot configuration
```

## Custom Composite Action

### `setup-env`

**Location**: `.github/actions/setup-env/action.yaml`

A reusable composite action that handles the common setup steps for most workflows.

**Purpose**:

- Installs `uv` package manager
- Sets up Python environment
- Installs dependencies using invoke tasks
- Shows environment configuration and dependency tree

**Inputs**:

- `python-version`: Python version to use (default: `3.13`)
- `package-name`: Optional package name to install
- `package-version`: Optional package version to install
- `install-args`: Optional arguments when installing dependencies (e.g., `--docs-deps`,
  `--no-optional-deps`)

**Usage**:

```yaml
- name: Setup python environment
  uses: ./.github/actions/setup-env
  with:
    python-version: '3.12'
    install-args: '--docs-deps'
```

## Main Workflows

### CI Workflow (`ci.yaml`)

**Triggers**:

- Pull requests to `main` branch
- Pushes to `main` branch
- Manual workflow dispatch

**Purpose**: Main continuous integration workflow that orchestrates all quality checks.

**Jobs**: Calls the following reusable workflows:

- `build` - Package build tests
- `coverage` - Code coverage reporting
- `cyclic-imports` - Cyclic import detection
- `doctest` - Documentation tests
- `format` - Code formatting checks
- `pre-commit` - Pre-commit hook validation
- `test` - Unit and integration tests
- `test-deps` - Dependency compatibility tests

### Build Tests (`build.yaml`)

**Triggers**:

- Called by other workflows (`workflow_call`)
- Manual workflow dispatch

**Purpose**: Validates that the package can be built and installed correctly across different distribution types and configurations.

**Jobs**:

1. **build**:
    - Uses a matrix strategy to test:
        - Distribution types: `sdist` (source distribution) and `wheel`
        - Extras: base installation (no extras) and `all` optional dependencies
    - Builds the package using `uv build`
    - Checks that `py.typed` is present in the distribution
    - Installs the built package (with or without extras)
    - Validates the build with `twine check`
    - Checks package metadata
    - Checks dependency tree
    - Verifies basic import works
    - Validates package version is not `0.0.0`
    - Verifies that `pyright` recognizes the package as typed via type checking

### Test Workflow (`test.yaml`)

**Triggers**:

- Called by other workflows (`workflow_call`)
- Manual workflow dispatch

**Purpose**: Runs comprehensive test suite across multiple Python versions and operating systems.

**Jobs**:

1. **all**:
    - Runs on multiple OS: Ubuntu (latest, 24.04, 22.04, ARM), macOS (latest, 26, 15, 14, Intel)
    - Tests Python versions: 3.10, 3.11, 3.12, 3.13, 3.14 (including free-threaded builds)
    - Runs unit and integration tests with coverage

2. **min**:
    - Same matrix as `all` but with minimal dependencies (`--no-optional-deps`)
    - Ensures the package works without optional dependencies

### Coverage Workflow (`coverage.yaml`)

**Triggers**:

- Called by other workflows (`workflow_call`)
- Manual workflow dispatch

**Purpose**: Generates code coverage reports and uploads them to Codecov.

**Jobs**:

- Runs unit and integration tests with coverage
- Uploads coverage data to Codecov (only for the main repository)
- Failures in uploading to Codecov do not fail the workflow

### Documentation Workflows

#### Stable Documentation (`docs.yaml`)

**Triggers**:

- Version tags (`v*.*.*`)
- Manual workflow dispatch

**Purpose**: Builds and deploys documentation for stable releases.

**Jobs**:

- Sets up environment with documentation dependencies
- Configures git for the docs bot
- Publishes documentation using `mike` (documentation versioning tool)

#### Development Documentation (`docs-dev.yaml`)

**Triggers**:

- Pushes to `main` branch
- Manual workflow dispatch

**Purpose**: Builds and deploys documentation for the development version.

### PyPI Publishing (`pypi.yaml`)

**Triggers**:

- Version tags (`v*.*.*`)
- Manual workflow dispatch

**Purpose**: Publishes the package to PyPI.

**Jobs**:

- Installs `uv` and `invoke`
- Runs the publish command with PyPI token from secrets

### Additional Workflows

#### Nightly Tests (`nightly-tests.yaml`)

**Triggers**:

- Scheduled daily at 9:10 AM UTC
- Manual workflow dispatch

**Purpose**: Runs the full test suite nightly to catch issues early.

#### Nightly Package Tests (`nightly-package.yaml`)

**Triggers**:

- Scheduled daily at 10:30 AM UTC
- Manual workflow dispatch

**Purpose**: Tests package installation and imports nightly.

#### Pre-commit (`pre-commit.yaml`)

**Triggers**:

- Called by other workflows
- Manual workflow dispatch

**Purpose**: Runs pre-commit hooks to validate code style and quality.

#### Pre-commit Auto-update (`pre-commit-autoupdate.yaml`)

**Triggers**:

- Scheduled weekly on Sundays at 5:10 AM UTC
- Manual workflow dispatch

**Purpose**: Automatically updates pre-commit hook versions.

#### Format Check (`format.yaml`)

**Triggers**:

- Called by other workflows
- Manual workflow dispatch

**Purpose**: Checks code formatting using Black and Ruff.

#### Cyclic Imports (`cyclic-imports.yaml`)

**Triggers**:

- Called by other workflows
- Manual workflow dispatch

**Purpose**: Detects circular import dependencies.

#### Doctest (`doctest.yaml`)

**Triggers**:

- Called by other workflows
- Manual workflow dispatch

**Purpose**: Runs doctests to validate code examples in documentation.

#### Test Dependencies (`test-deps.yaml`)

**Triggers**:

- Called by other workflows
- Manual workflow dispatch

**Purpose**: Tests the package with different versions of dependencies.

#### Dependency Review (`dependency-review.yml`)

**Triggers**:

- Pull requests

**Purpose**: Reviews dependency changes for security vulnerabilities.

#### Update Dependencies (`update-deps.yaml`)

**Triggers**:

- Scheduled weekly on Sundays at 5:10 AM UTC
- Manual workflow dispatch

**Purpose**: Creates pull requests to update dependencies.

#### Cancel Stale Queued Runs (`cancel_stale_queued_runs.yaml`)

**Triggers**:

- Push to any branch
- Pull requests

**Purpose**: Cancels outdated workflow runs to save CI resources.

#### Generate Package Versions (`generate-package-versions.yaml`)

**Triggers**:

- Manual workflow dispatch

**Purpose**: Generates version compatibility documentation.

#### Get Versions (`get-versions.yaml`)

**Triggers**:

- Manual workflow dispatch

**Purpose**: Retrieves version information from the project.

## Workflow Permissions

Most workflows use minimal permissions following the principle of least privilege:

- `contents: read` - Default for read-only operations
- `contents: write` - Only for workflows that need to push changes (e.g., documentation deployment)

## Secrets Used

The following secrets are configured in the repository settings:

- `CODECOV_TOKEN` - For uploading coverage reports to Codecov
- `PYPI_TOKEN` - For publishing packages to PyPI
- `CI_BOT_EMAIL` - Email address for the CI bot (used for documentation deployment)

## Best Practices

1. **Reusable Workflows**: Common workflows are defined with `workflow_call` trigger so they can be
   reused by other workflows.

2. **Matrix Testing**: The test workflows use matrix strategies to test across multiple Python
   versions and operating systems.

3. **Composite Actions**: The `setup-env` action centralizes common setup steps, reducing
   duplication.

4. **Timeouts**: All jobs have reasonable timeout limits (typically 10 minutes) to prevent hung
   workflows.

5. **Fail-Fast**: Most matrix jobs use `fail-fast: false` to continue testing other combinations
   even if one fails.

6. **Version Pinning**: GitHub Actions are pinned to specific major versions (e.g., `@v6`, `@v7`)
   for stability.

7. **Minimal Changes**: Workflows are designed to make minimal assumptions about the environment and
   clean up after themselves.

## Dependabot Configuration

The repository uses Dependabot to automatically update GitHub Actions dependencies. The
configuration is in `.github/dependabot.yml`.

## Issue and Pull Request Templates

The repository provides templates to help contributors submit well-structured issues and pull
requests:

- **Bug Report**: `.github/ISSUE_TEMPLATE/bug-report.yml`
- **Feature Request**: `.github/ISSUE_TEMPLATE/feature-request.yml`
- **Pull Request Template**: `.github/PULL_REQUEST_TEMPLATE.md`

## Troubleshooting Workflow Issues

### Common Issues and Solutions

#### Workflow Fails with "Permission denied"

**Cause**: Insufficient permissions in the workflow configuration.

**Solution**: Verify that the workflow has the necessary permissions. For example, workflows that push changes need `contents: write`.

#### Workflow Hangs or Times Out

**Cause**: A step is taking longer than expected or is waiting for input.

**Solution**:
- Check the timeout settings in the workflow (default is 10 minutes for most jobs)
- Review job logs to identify which step is hanging
- Consider increasing the timeout if the step legitimately needs more time

#### Matrix Job Failures

**Cause**: One configuration in the matrix is failing while others pass.

**Solution**:
- Check the specific configuration that failed (Python version, OS, etc.)
- Review the job logs for that specific matrix combination
- Test locally with the same configuration if possible

#### Secret Not Available

**Cause**: Required secret is not configured in repository settings.

**Solution**:
- Go to repository Settings > Secrets and variables > Actions
- Add the required secret (e.g., `CODECOV_TOKEN`, `PYPI_TOKEN`)

#### Dependabot PR Fails

**Cause**: Dependency update breaks tests or compatibility.

**Solution**:
- Review the Dependabot PR description for breaking changes
- Check the failing test logs
- Either fix the compatibility issue or close the PR if the update is not needed

### Manually Triggering Workflows

Most workflows can be manually triggered using `workflow_dispatch`:

1. Go to the "Actions" tab in GitHub
2. Select the workflow you want to run
3. Click "Run workflow"
4. Select the branch and any required inputs
5. Click "Run workflow" button

### Viewing Workflow Logs

To view detailed logs for a workflow run:

1. Go to the "Actions" tab
2. Click on the workflow run
3. Click on a specific job to see its steps
4. Click on a step to expand its logs
5. Use the search box to filter log output

### Local Testing

Before pushing changes, you can test workflows locally:

```bash
# Run tests locally
inv unit-test --cov
inv integration-test --cov

# Check code formatting
inv check-format

# Check types
inv check-types

# Run pre-commit hooks
pre-commit run --all-files

# Check markdown documentation
./dev/check_markdown.sh
```

## Workflow Usage Examples

### Running Build Validation Locally

To replicate what the build workflow does:

```bash
# Build source distribution
uv build --sdist

# Build wheel
uv build --wheel

# Install from built wheel
uv pip install dist/metriclab-*.whl

# Validate package metadata
./dev/package/check_metadata.sh

# Check dependency tree
./dev/package/check_dependency_tree.sh

# Validate type annotations
./dev/package/check_type.sh

# Run custom checks
./dev/package/custom_checks.sh
```

### Testing with Different Python Versions

The test workflow tests multiple Python versions. To test locally with different versions:

```bash
# Using pyenv to test with Python 3.11
pyenv install 3.11
pyenv local 3.11
python -m venv .venv-3.11
source .venv-3.11/bin/activate
uv pip install -e .
inv unit-test

# Test with minimal dependencies
uv pip install -e . --no-optional-deps
inv unit-test
```


## Maintenance Tasks

### Updating Pre-commit Hooks

Happens automatically weekly via `pre-commit-autoupdate.yaml`, but can be done manually:

```bash
pre-commit autoupdate
pre-commit run --all-files
git commit -am "Update pre-commit hooks"
```

### Updating Dependencies

Happens automatically weekly via `update-deps.yaml`, but can be done manually:

```bash
inv update
inv install --docs-deps
inv unit-test --cov
```

### Regenerating Package Version Configuration

Happens automatically weekly via `generate-package-versions.yaml`, but can be done manually:

```bash
python dev/generate_versions.py
git commit -am "Generate package versions"
```

## Contributing

For detailed information about contributing to the project, including how to run tests locally and
development setup,
see [CONTRIBUTING.md](https://github.com/durandtibo/metriclab/blob/main/.github/CONTRIBUTING.md).

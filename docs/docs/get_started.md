# Get Started

It is highly recommended to install in
a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
to keep your system in order.

## Installing with `uv` (recommended)

The following command installs the latest stable version of the library:

```shell
uv pip install metriclab
```

To install the latest development version from GitHub:

```shell
uv pip install git+https://github.com/durandtibo/metriclab.git
```

To install a specific version:

```shell
uv pip install metriclab==0.3.0
```

## Installing with `pip`

The following command installs the latest stable version of the library:

```shell
pip install metriclab
```

To install the latest development version from GitHub:

```shell
pip install git+https://github.com/durandtibo/metriclab.git
```

To install a specific version:

```shell
pip install metriclab==0.3.0
```

## Verifying Installation

After installation, you can verify that metriclab is correctly installed by running:

```shell
python -c "import metriclab; print(metriclab.__version__)"
```

Or try a simple example:

```python
from metriclab.functional import accuracy

result = accuracy(y_true=[1, 0, 1, 1], y_pred=[1, 1, 1, 0])
print(result.accuracy)  # Output: 0.5
```

## Installing from source

To install `metriclab` from source, you can follow the steps below.

### Prerequisites

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management. Please refer to
the [uv installation documentation](https://docs.astral.sh/uv/getting-started/installation/) for
installation instructions.

### Clone the Repository

```shell
git clone git@github.com:durandtibo/metriclab.git
cd metriclab
```

### Create a Virtual Environment

It is recommended to create a Python 3.10+ virtual environment:

```shell
make setup-venv
```

This command creates a virtual environment using `uv` and installs all dependencies including
development tools.

Alternatively, you can create a conda virtual environment:

```shell
make conda
conda activate metriclab
make install
```

### Install Dependencies

To install only the core dependencies:

```shell
make install
```

To install all dependencies including documentation tools:

```shell
make install-all
```

### Verify Installation

You can test the installation with the following command:

```shell
make unit-test-cov
```

This will run the test suite with coverage reporting.

## Development Setup

If you plan to contribute to metriclab, please also install the development tools.

Recommended:

```shell
make install-all
```

This installs the project in editable mode together with optional, development,
and documentation dependencies.

If you prefer to use `uv` directly:

```shell
uv sync --frozen --all-extras --group dev --group docs
uv pip install -e .
```

Then install the pre-commit hooks:

```shell
pre-commit install
```

See [CONTRIBUTING.md](https://github.com/durandtibo/metriclab/blob/main/CONTRIBUTING.md) for
more information about contributing.

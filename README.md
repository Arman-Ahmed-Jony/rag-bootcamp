# RAG Bootcamp

A Python project for learning and implementing Retrieval-Augmented Generation (RAG) systems.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.13+**: Download from [python.org](https://www.python.org/downloads/) or use a version manager like `pyenv`
- **uv**: A fast Python package manager and project manager

## Installation

### 1. Install uv

If you don't have `uv` installed, install it using one of these methods:

**Using pip:**
```bash
pip install uv
```

**Using curl (macOS/Linux):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Using PowerShell (Windows):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Using Homebrew (macOS):**
```bash
brew install uv
```

### 2. Clone the Repository

```bash
git clone https://github.com/Arman-Ahmed-Jony/rag-bootcamp.git
cd rag-bootcamp
```

### 3. Set up the Project

Initialize the project and create a virtual environment:

```bash
# Initialize the project (if not already done)
uv init

# Create and activate virtual environment
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate
```

### 4. Install Dependencies

```bash
# Install project dependencies
uv sync

# Install additional development dependencies
uv add ipykernel

# Or install dependencies in development mode
uv pip install -e .
```

## Running the Project

```bash
# Make sure virtual environment is activated
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Run the main script
python main.py
```

## Development Workflow

### Adding Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Add Jupyter kernel support (for notebooks)
uv add ipykernel

# Add a specific version
uv add "package-name>=1.0.0,<2.0.0"
```

### Updating Dependencies

```bash
# Update all dependencies
uv sync --upgrade

# Update a specific package
uv add package-name@latest
```

### Running Scripts

```bash
# Run any Python script
uv run python script_name.py

# Run with specific Python version
uv run --python 3.13 python script_name.py

# Run Jupyter notebooks (after installing ipykernel)
jupyter notebook

# Or run JupyterLab
jupyter lab
```

## Project Structure

```
rag-bootcamp/
├── .venv/                 # Virtual environment (created by uv)
├── .python-version        # Python version specification
├── .gitignore            # Git ignore rules
├── main.py               # Main application entry point
├── pyproject.toml        # Project configuration and dependencies
└── README.md             # This file
```

## Troubleshooting

### Virtual Environment Issues

If you encounter issues with the virtual environment:

```bash
# Remove existing virtual environment
rm -rf .venv

# Recreate virtual environment
uv venv

# Reactivate
source .venv/bin/activate
```

### Python Version Issues

If you need to use a different Python version:

```bash
# Check available Python versions
uv python list

# Install a specific Python version
uv python install 3.13

# Use specific Python version for the project
uv venv --python 3.13
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Commit your changes: `git commit -m 'Add some feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

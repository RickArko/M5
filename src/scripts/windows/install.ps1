$ErrorActionPreference = "Stop"

Write-Host "Setting up M5 project with uv..."

Write-Host "Creating virtual environment and installing dependencies (editable)..."
uv sync --all-groups

Write-Host "Installing Jupyter kernel..."
uv run python -m ipykernel install --user --name m5 --display-name "m5"

Write-Host "Running data generation..."
uv run python src/generate_data.py

Write-Host "Running data processing..."
uv run python src/process.py

Write-Host "Setup complete!"

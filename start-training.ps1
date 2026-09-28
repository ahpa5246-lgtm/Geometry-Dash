$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Virtual environment missing. Run .\setup.ps1 first."
}

& .\.venv\Scripts\Activate.ps1
python -m gd_ai.train

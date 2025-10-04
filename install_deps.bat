@echo off
echo "Activating virtual environment..."
call .\.venv\Scripts\activate.bat

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
python -m pip install -r backend\requirements.txt

echo "Installing openai-whisper from git..."
python -m pip install git+https://github.com/openai/whisper.git

echo "Installation complete. You can now start the servers."
pause 
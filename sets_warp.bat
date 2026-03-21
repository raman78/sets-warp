@echo off
:: SETS.bat — single entry point for SETS (Windows)
:: bootstrap.py handles venv creation, dependency installation and launch.

cd /d "%~dp0"
if errorlevel 1 (
    echo [Error] Cannot change to script directory: %~dp0
    exit /b 1
)

set PYTHON=

:: Prefer Python 3.13 via Windows py launcher
py -3.13 --version >nul 2>&1
if not errorlevel 1 ( set PYTHON=py -3.13 & goto :found )

:: Try py launcher with other versions
py -3.14 --version >nul 2>&1
if not errorlevel 1 ( set PYTHON=py -3.14 & goto :found )

py -3.12 --version >nul 2>&1
if not errorlevel 1 ( set PYTHON=py -3.12 & goto :found )

:: Try generic py launcher
py --version >nul 2>&1
if not errorlevel 1 ( set PYTHON=py & goto :found )

:: Try direct executables
python3.13 --version >nul 2>&1
if not errorlevel 1 ( set PYTHON=python3.13 & goto :found )

python3 --version >nul 2>&1
if not errorlevel 1 ( set PYTHON=python3 & goto :found )

python --version >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; sys.exit(0 if sys.version_info.major==3 else 1)" >nul 2>&1
    if not errorlevel 1 ( set PYTHON=python & goto :found )
)

echo [Error] No Python 3 installation found.
echo         Please install Python 3.13 or later: https://www.python.org/downloads/
echo         Make sure to check "Add Python to PATH" during installation.
pause
exit /b 1

:found
%PYTHON% bootstrap.py %*

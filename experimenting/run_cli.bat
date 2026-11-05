@echo off
setlocal DisableDelayedExpansion
if defined EXPERIMENT_PYTHON set "PYTHON=%EXPERIMENT_PYTHON%"
if defined PYTHON goto custom
python -c "import sys; sys.exit(sys.version_info < (3,10))" >nul 2>nul
if not errorlevel 1 goto python
py -3 -c "import sys; sys.exit(sys.version_info < (3,10))" >nul 2>nul
if not errorlevel 1 goto launcher
echo Python 3.10+ diperlukan. Aktifkan Conda/venv atau set PYTHON. 1>&2
exit /b 2
:custom
"%PYTHON%" "%~dp0..\scripts\experiment_cli.py" %* --profile experimenting
exit /b %errorlevel%
:python
python "%~dp0..\scripts\experiment_cli.py" %* --profile experimenting
exit /b %errorlevel%
:launcher
py -3 "%~dp0..\scripts\experiment_cli.py" %* --profile experimenting
exit /b %errorlevel%

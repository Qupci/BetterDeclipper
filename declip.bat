@echo off
rem BetterDeclipper launcher: uses the project venv (GPU build of torch) if present.
set "BD_HOME=%~dp0"
set "PYTHONPATH=%BD_HOME%;%PYTHONPATH%"
if exist "%BD_HOME%.venv\Scripts\python.exe" (
  "%BD_HOME%.venv\Scripts\python.exe" -m betterdeclipper %*
) else (
  python -m betterdeclipper %*
)

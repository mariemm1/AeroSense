@echo off
setlocal

REM --- Go to project root: from Data\ExtractData -> back to AeroSense ---
cd /d "%~dp0\..\.."

REM --- Activate the SAME virtualenv ---
call Data\my_env\Scripts\activate.bat

REM --- Run Sentinel-3 LST pipeline ---
python Data\ExtractData\s3_pipeline.py ^
  --regions "tunisia,ariana,tozeur,manouba,siliana" ^
  --top 3

REM --- Deactivate venv (optional) ---
deactivate

endlocal

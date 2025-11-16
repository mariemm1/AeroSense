@echo off
setlocal

REM --- Go to project root: from Data\ExtractData -> back to AeroSense ---
cd /d "%~dp0\..\.."

REM --- Activate the SAME virtualenv where we installed numpy etc. ---
call Data\my_env\Scripts\activate.bat

REM --- Run Sentinel-5P pipeline ---
REM   Change regions if you want other areas.
python Data\ExtractData\s5p_pipeline.py ^
  --regions "tunisia,ariana,tozeur,manouba,siliana" ^
  --top 3

REM --- Deactivate venv (optional) ---
deactivate

endlocal

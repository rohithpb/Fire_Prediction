@echo off
setlocal

if "%~1"=="" (
    echo ======================================================
    echo       NASA Active Fire Risk Predictor
    echo ======================================================
    set /p P_DATE="Enter date (YYYY-MM-DD) [default 2026-03-15]: "
    if "!P_DATE!"=="" set P_DATE=2026-03-15
    if "%P_DATE%"=="" set P_DATE=2026-03-15

    set /p P_LAT="Enter latitude (8.1 to 14.6) [default 10.5]: "
    if "!P_LAT!"=="" set P_LAT=10.5
    if "%P_LAT%"=="" set P_LAT=10.5

    set /p P_LON="Enter longitude (74.4 to 77.9) [default 76.2]: "
    if "!P_LON!"=="" set P_LON=76.2
    if "%P_LON%"=="" set P_LON=76.2

    echo.
    echo Running prediction...
    echo.
    .\.venv\Scripts\python.exe predict.py --date %P_DATE% --lat %P_LAT% --lon %P_LON% --model gru
    echo.
    pause
) else (
    .\.venv\Scripts\python.exe predict.py %*
)

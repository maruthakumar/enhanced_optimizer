@echo off
REM ============================================================================
REM HeavyDB GPU Optimizer - Portfolio Management Template
REM Optimized for quality: 250 strategies, 8-15 minutes, 95%+ success rate
REM Expected fitness improvement: 20-35%
REM ============================================================================

setlocal enabledelayedexpansion
set "SERVER_IP=204.12.223.93"
set "SHARE_PATH=\\%SERVER_IP%\optimizer_share"
set "USERNAME=opt_admin"
set "PASSWORD=Chetti@123"

echo.
echo ============================================================================
echo HeavyDB GPU Multi-Algorithm Optimization Platform
echo PORTFOLIO MANAGEMENT OPTIMIZATION
echo ============================================================================
echo.
echo Expected Performance:
echo - Strategy Count: 250
echo - Execution Time: 8-15 minutes
echo - Success Rate: 95%+
echo - Fitness Improvement: 20-35%
echo.

REM Check if input file is provided
if "%~1"=="" (
    echo ERROR: Please provide an Excel file as input
    echo Usage: %0 "path\to\your\dataset.xlsx"
    echo Example: %0 "C:\Data\SENSEX_data.xlsx"
    pause
    exit /b 1
)

set "INPUT_FILE=%~1"
set "FILE_NAME=%~nx1"
set "JOB_ID=PORTFOLIO_%DATE:~-4,4%%DATE:~-10,2%%DATE:~-7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%"
set "JOB_ID=%JOB_ID: =0%"

echo Input File: %INPUT_FILE%
echo Job ID: %JOB_ID%
echo.

REM Check if input file exists
if not exist "%INPUT_FILE%" (
    echo ERROR: Input file not found: %INPUT_FILE%
    pause
    exit /b 1
)

echo [1/5] Connecting to HeavyDB Optimizer Server...
net use P: %SHARE_PATH% %PASSWORD% /user:%USERNAME% /persistent:no >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to connect to server. Please check network connection.
    pause
    exit /b 1
)
echo ✓ Connected to server successfully

echo.
echo [2/5] Uploading dataset file...
copy "%INPUT_FILE%" "P:\input\%FILE_NAME%" >nul
if errorlevel 1 (
    echo ERROR: Failed to upload file to server
    net use P: /delete >nul 2>&1
    pause
    exit /b 1
)
echo ✓ Dataset uploaded: %FILE_NAME%

echo.
echo [3/5] Creating optimization job configuration...
(
echo {
echo   "job_id": "%JOB_ID%",
echo   "input_file": "%FILE_NAME%",
echo   "strategy_count": 250,
echo   "strategy": "ensemble",
echo   "min_size": 20,
echo   "max_size": 50,
echo   "timeout": 900,
echo   "algorithms": ["all"],
echo   "fitness_metric": "ratio",
echo   "optimization_type": "portfolio",
echo   "created_at": "%DATE:~-4,4%-%DATE:~-10,2%-%DATE:~-7,2%T%TIME:~0,2%:%TIME:~3,2%:%TIME:~6,2%Z"
echo }
) > "P:\input\%JOB_ID%_config.json"
echo ✓ Job configuration created

echo.
echo [4/5] Starting optimization process...
echo Job submitted to processing queue. Monitoring progress...
echo This may take 8-15 minutes for portfolio optimization...

set "TIMEOUT_COUNT=0"
set "MAX_TIMEOUT=900"

:check_progress
timeout /t 5 >nul
set /a TIMEOUT_COUNT+=5

if exist "P:\output\%JOB_ID%_status.json" (
    for /f "tokens=*" %%i in ('type "P:\output\%JOB_ID%_status.json" 2^>nul ^| findstr "completed"') do (
        if not "%%i"=="" goto job_completed
    )
    for /f "tokens=*" %%i in ('type "P:\output\%JOB_ID%_status.json" 2^>nul ^| findstr "failed"') do (
        if not "%%i"=="" goto job_failed
    )
)

set /a MINUTES=%TIMEOUT_COUNT%/60
echo Progress: %MINUTES% minutes elapsed (Expected: 8-15 minutes)

if %TIMEOUT_COUNT% lss %MAX_TIMEOUT% (
    goto check_progress
)

echo WARNING: Optimization taking longer than expected (15+ minutes)
echo This may indicate a very large dataset or high system load
echo Continue waiting? (Y/N)
set /p CONTINUE=
if /i "%CONTINUE%"=="Y" (
    set "MAX_TIMEOUT=1800"
    goto check_progress
)
goto job_timeout

:job_completed
set /a FINAL_MINUTES=%TIMEOUT_COUNT%/60
echo ✓ Optimization completed successfully in %FINAL_MINUTES% minutes!

echo.
echo [5/5] Downloading results...
if exist "P:\output\%JOB_ID%_results.json" (
    copy "P:\output\%JOB_ID%_results.json" "%~dp0%JOB_ID%_results.json" >nul
    echo ✓ Results downloaded: %JOB_ID%_results.json
) else (
    echo WARNING: Results file not found
)

if exist "P:\output\%JOB_ID%_status.json" (
    copy "P:\output\%JOB_ID%_status.json" "%~dp0%JOB_ID%_status.json" >nul
    echo ✓ Status downloaded: %JOB_ID%_status.json
)

REM Check for additional result files
for %%f in (P:\output\%JOB_ID%_*.xlsx P:\output\%JOB_ID%_*.png P:\output\%JOB_ID%_*.csv) do (
    if exist "%%f" (
        copy "%%f" "%~dp0" >nul
        echo ✓ Downloaded: %%~nxf
    )
)

echo.
echo ============================================================================
echo PORTFOLIO MANAGEMENT OPTIMIZATION COMPLETED
echo ============================================================================
echo Job ID: %JOB_ID%
echo Execution Time: %FINAL_MINUTES% minutes
echo Results available in current directory
echo.
type "%~dp0%JOB_ID%_results.json" 2>nul
echo.
goto cleanup

:job_failed
echo ✗ Optimization failed. Check status file for details.
if exist "P:\output\%JOB_ID%_status.json" (
    copy "P:\output\%JOB_ID%_status.json" "%~dp0%JOB_ID%_error.json" >nul
    echo Error details saved to: %JOB_ID%_error.json
)
goto cleanup

:job_timeout
echo ✗ Optimization timed out. Job may still be processing on server.
echo Check server status or try again later.
goto cleanup

:cleanup
echo.
echo Cleaning up connection...
net use P: /delete >nul 2>&1
echo ✓ Disconnected from server

echo.
echo Press any key to exit...
pause >nul
exit /b 0

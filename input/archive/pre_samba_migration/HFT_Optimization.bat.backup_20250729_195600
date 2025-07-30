@echo off
REM ============================================================================
REM HeavyDB GPU Optimizer - High-Frequency Trading Template
REM Optimized for speed: 100 strategies, 2-4 minutes, 98%+ success rate
REM Expected fitness improvement: 15-25%
REM ============================================================================

setlocal enabledelayedexpansion
set "SERVER_IP=204.12.223.93"
set "SHARE_PATH=\\%SERVER_IP%\optimizer_share"
set "USERNAME=opt_admin"
set "PASSWORD=Chetti@123"
set "SSH_KEY=heavydb_optimizer_key.ppk"

echo.
echo ============================================================================
echo HeavyDB GPU Multi-Algorithm Optimization Platform
echo HIGH-FREQUENCY TRADING OPTIMIZATION
echo ============================================================================
echo.
echo Expected Performance:
echo - Strategy Count: 100
echo - Execution Time: 2-4 minutes
echo - Success Rate: 98%+
echo - Fitness Improvement: 15-25%
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
set "JOB_ID=HFT_%DATE:~-4,4%%DATE:~-10,2%%DATE:~-7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%"
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
net use H: %SHARE_PATH% %PASSWORD% /user:%USERNAME% /persistent:no >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to connect to server. Please check network connection.
    pause
    exit /b 1
)
echo ✓ Connected to server successfully

echo.
echo [2/5] Uploading dataset file...
copy "%INPUT_FILE%" "H:\input\%FILE_NAME%" >nul
if errorlevel 1 (
    echo ERROR: Failed to upload file to server
    net use H: /delete >nul 2>&1
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
echo   "strategy_count": 100,
echo   "strategy": "parallel",
echo   "min_size": 10,
echo   "max_size": 25,
echo   "timeout": 240,
echo   "algorithms": ["hc", "sa"],
echo   "fitness_metric": "ratio",
echo   "optimization_type": "hft",
echo   "created_at": "%DATE:~-4,4%-%DATE:~-10,2%-%DATE:~-7,2%T%TIME:~0,2%:%TIME:~3,2%:%TIME:~6,2%Z"
echo }
) > "H:\input\%JOB_ID%_config.json"
echo ✓ Job configuration created

echo.
echo [4/5] Starting optimization process...
echo Job submitted to processing queue. Monitoring progress...

set "TIMEOUT_COUNT=0"
set "MAX_TIMEOUT=150"

:check_progress
timeout /t 2 >nul
set /a TIMEOUT_COUNT+=2

if exist "H:\output\%JOB_ID%_status.json" (
    for /f "tokens=*" %%i in ('type "H:\output\%JOB_ID%_status.json" 2^>nul ^| findstr "completed"') do (
        if not "%%i"=="" goto job_completed
    )
    for /f "tokens=*" %%i in ('type "H:\output\%JOB_ID%_status.json" 2^>nul ^| findstr "failed"') do (
        if not "%%i"=="" goto job_failed
    )
)

if %TIMEOUT_COUNT% lss %MAX_TIMEOUT% (
    echo Progress: %TIMEOUT_COUNT%/%MAX_TIMEOUT% seconds elapsed...
    goto check_progress
)

echo WARNING: Optimization taking longer than expected (5+ minutes)
echo This may indicate a large dataset or system load
echo Continue waiting? (Y/N)
set /p CONTINUE=
if /i "%CONTINUE%"=="Y" (
    set "MAX_TIMEOUT=300"
    goto check_progress
)
goto job_timeout

:job_completed
echo ✓ Optimization completed successfully!

echo.
echo [5/5] Downloading results...
if exist "H:\output\%JOB_ID%_results.json" (
    copy "H:\output\%JOB_ID%_results.json" "%~dp0%JOB_ID%_results.json" >nul
    echo ✓ Results downloaded: %JOB_ID%_results.json
) else (
    echo WARNING: Results file not found
)

if exist "H:\output\%JOB_ID%_status.json" (
    copy "H:\output\%JOB_ID%_status.json" "%~dp0%JOB_ID%_status.json" >nul
    echo ✓ Status downloaded: %JOB_ID%_status.json
)

echo.
echo ============================================================================
echo HIGH-FREQUENCY TRADING OPTIMIZATION COMPLETED
echo ============================================================================
echo Job ID: %JOB_ID%
echo Results available in current directory
echo.
type "%~dp0%JOB_ID%_results.json" 2>nul
echo.
goto cleanup

:job_failed
echo ✗ Optimization failed. Check status file for details.
if exist "H:\output\%JOB_ID%_status.json" (
    copy "H:\output\%JOB_ID%_status.json" "%~dp0%JOB_ID%_error.json" >nul
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
net use H: /delete >nul 2>&1
echo ✓ Disconnected from server

echo.
echo Press any key to exit...
pause >nul
exit /b 0

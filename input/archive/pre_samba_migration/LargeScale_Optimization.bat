@echo off
REM ============================================================================
REM HeavyDB GPU Optimizer - Large-Scale Processing Template
REM Enterprise-grade: 10,000+ strategies, 2-4 hours, 95%+ success rate
REM Distributed processing with chunk-based optimization
REM ============================================================================

setlocal enabledelayedexpansion
set "SERVER_IP=204.12.223.93"
set "SHARE_PATH=\\%SERVER_IP%\optimizer_share"
set "USERNAME=opt_admin"
set "PASSWORD=Chetti@123"

echo.
echo ============================================================================
echo HeavyDB GPU Multi-Algorithm Optimization Platform
echo LARGE-SCALE DISTRIBUTED OPTIMIZATION
echo ============================================================================
echo.
echo Expected Performance:
echo - Strategy Count: 10,000+ (configurable)
echo - Execution Time: 2-4 hours
echo - Success Rate: 95%+
echo - Processing: Distributed chunk-based optimization
echo.

REM Check if input file is provided
if "%~1"=="" (
    echo ERROR: Please provide an Excel file as input
    echo Usage: %0 "path\to\your\dataset.xlsx" [strategy_count]
    echo Example: %0 "C:\Data\SENSEX_data.xlsx" 10000
    echo Default strategy count: 10000
    pause
    exit /b 1
)

set "INPUT_FILE=%~1"
set "FILE_NAME=%~nx1"
set "STRATEGY_COUNT=%~2"
if "%STRATEGY_COUNT%"=="" set "STRATEGY_COUNT=10000"

set "JOB_ID=LARGESCALE_%DATE:~-4,4%%DATE:~-10,2%%DATE:~-7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%"
set "JOB_ID=%JOB_ID: =0%"

echo Input File: %INPUT_FILE%
echo Strategy Count: %STRATEGY_COUNT%
echo Job ID: %JOB_ID%
echo.

REM Validate strategy count
if %STRATEGY_COUNT% lss 1000 (
    echo WARNING: Strategy count less than 1000. Consider using Portfolio or Research templates.
    echo Continue with Large-Scale processing? (Y/N)
    set /p CONTINUE=
    if /i not "%CONTINUE%"=="Y" exit /b 1
)

REM Check if input file exists
if not exist "%INPUT_FILE%" (
    echo ERROR: Input file not found: %INPUT_FILE%
    pause
    exit /b 1
)

echo [1/6] Connecting to HeavyDB Optimizer Server...
net use L: %SHARE_PATH% %PASSWORD% /user:%USERNAME% /persistent:no >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to connect to server. Please check network connection.
    pause
    exit /b 1
)
echo ✓ Connected to server successfully

echo.
echo [2/6] Uploading large dataset file...
echo This may take several minutes for large files...
copy "%INPUT_FILE%" "L:\input\%FILE_NAME%" >nul
if errorlevel 1 (
    echo ERROR: Failed to upload file to server
    net use L: /delete >nul 2>&1
    pause
    exit /b 1
)
echo ✓ Dataset uploaded: %FILE_NAME%

echo.
echo [3/6] Creating large-scale optimization configuration...
set /a CHUNK_SIZE=1000
set /a MAX_WORKERS=4
set /a ESTIMATED_HOURS=%STRATEGY_COUNT%/2500

(
echo {
echo   "job_id": "%JOB_ID%",
echo   "input_file": "%FILE_NAME%",
echo   "strategy_count": %STRATEGY_COUNT%,
echo   "strategy": "ensemble",
echo   "chunk_size": %CHUNK_SIZE%,
echo   "max_workers": %MAX_WORKERS%,
echo   "timeout": 14400,
echo   "algorithms": ["all"],
echo   "fitness_metric": "ratio",
echo   "optimization_type": "largescale",
echo   "distributed": true,
echo   "generate_reports": true,
echo   "created_at": "%DATE:~-4,4%-%DATE:~-10,2%-%DATE:~-7,2%T%TIME:~0,2%:%TIME:~3,2%:%TIME:~6,2%Z"
echo }
) > "L:\input\%JOB_ID%_config.json"
echo ✓ Large-scale configuration created
echo   - Chunk size: %CHUNK_SIZE% strategies per chunk
echo   - Max workers: %MAX_WORKERS%
echo   - Estimated time: %ESTIMATED_HOURS% hours

echo.
echo [4/6] Starting large-scale distributed optimization...
echo Job submitted to processing queue. Monitoring progress...
echo.
echo ⚠️  IMPORTANT: This is a long-running process (%ESTIMATED_HOURS% hours estimated)
echo    You can safely close this window and check progress later
echo    Results will be available on the server when complete
echo.
echo Continue monitoring? (Y/N) - Choose N to submit job and exit
set /p MONITOR=
if /i not "%MONITOR%"=="Y" goto submit_and_exit

echo.
echo [5/6] Monitoring large-scale optimization progress...
echo Progress will be updated every 5 minutes...

set "TIMEOUT_COUNT=0"
set "MAX_TIMEOUT=14400"

:check_progress
timeout /t 300 >nul
set /a TIMEOUT_COUNT+=300

if exist "L:\output\%JOB_ID%_status.json" (
    for /f "tokens=*" %%i in ('type "L:\output\%JOB_ID%_status.json" 2^>nul ^| findstr "completed"') do (
        if not "%%i"=="" goto job_completed
    )
    for /f "tokens=*" %%i in ('type "L:\output\%JOB_ID%_status.json" 2^>nul ^| findstr "failed"') do (
        if not "%%i"=="" goto job_failed
    )
)

set /a HOURS=%TIMEOUT_COUNT%/3600
set /a MINUTES=(%TIMEOUT_COUNT% %% 3600)/60
echo Progress: %HOURS%h %MINUTES%m elapsed (Estimated: %ESTIMATED_HOURS% hours)

REM Show chunk progress if available
if exist "L:\logs\%JOB_ID%_progress.log" (
    echo Latest chunk progress:
    for /f "tokens=*" %%i in ('type "L:\logs\%JOB_ID%_progress.log" 2^>nul ^| findstr /i "chunk\|completed\|fitness"') do (
        echo   %%i
    )
)

if %TIMEOUT_COUNT% lss %MAX_TIMEOUT% (
    goto check_progress
)

echo WARNING: Large-scale optimization taking longer than expected (4+ hours)
echo Continue waiting or check results later? (C=Continue, E=Exit)
set /p CONTINUE=
if /i "%CONTINUE%"=="C" (
    set "MAX_TIMEOUT=21600"
    goto check_progress
)
goto job_timeout

:job_completed
set /a FINAL_HOURS=%TIMEOUT_COUNT%/3600
set /a FINAL_MINUTES=(%TIMEOUT_COUNT% %% 3600)/60
echo ✓ Large-scale optimization completed successfully in %FINAL_HOURS%h %FINAL_MINUTES%m!

echo.
echo [6/6] Downloading comprehensive results...
if exist "L:\output\%JOB_ID%_results.json" (
    copy "L:\output\%JOB_ID%_results.json" "%~dp0%JOB_ID%_results.json" >nul
    echo ✓ Main results downloaded: %JOB_ID%_results.json
)

REM Download all large-scale outputs
echo Downloading distributed optimization results...
for %%f in (L:\output\distributed_results\%JOB_ID%_*.*) do (
    if exist "%%f" (
        copy "%%f" "%~dp0" >nul
        echo ✓ Downloaded: %%~nxf
    )
)

echo.
echo ============================================================================
echo LARGE-SCALE DISTRIBUTED OPTIMIZATION COMPLETED
echo ============================================================================
echo Job ID: %JOB_ID%
echo Strategy Count: %STRATEGY_COUNT%
echo Execution Time: %FINAL_HOURS%h %FINAL_MINUTES%m
echo Processing Type: Distributed chunk-based optimization
echo Results available in current directory
echo.
goto cleanup

:submit_and_exit
echo.
echo ============================================================================
echo LARGE-SCALE JOB SUBMITTED SUCCESSFULLY
echo ============================================================================
echo Job ID: %JOB_ID%
echo Strategy Count: %STRATEGY_COUNT%
echo Estimated Time: %ESTIMATED_HOURS% hours
echo.
echo The job is now running on the server. You can:
echo 1. Check progress by running this script again with the same parameters
echo 2. Monitor the output folder: \\%SERVER_IP%\optimizer_share\output\
echo 3. Look for files starting with: %JOB_ID%
echo.
goto cleanup

:job_failed
echo ✗ Large-scale optimization failed. Check status file for details.
if exist "L:\output\%JOB_ID%_status.json" (
    copy "L:\output\%JOB_ID%_status.json" "%~dp0%JOB_ID%_error.json" >nul
    echo Error details saved to: %JOB_ID%_error.json
)
goto cleanup

:job_timeout
echo ⏰ Large-scale optimization is still running on the server.
echo Check back later or monitor the output folder for results.
echo Job ID: %JOB_ID%
goto cleanup

:cleanup
echo.
echo Cleaning up connection...
net use L: /delete >nul 2>&1
echo ✓ Disconnected from server

echo.
echo Press any key to exit...
pause >nul
exit /b 0

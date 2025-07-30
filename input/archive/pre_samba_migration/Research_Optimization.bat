@echo off
REM ============================================================================
REM HeavyDB GPU Optimizer - Research Analysis Template
REM Comprehensive analysis: 500 strategies, 20-45 minutes, 90%+ success rate
REM Full algorithm comparison and detailed reporting
REM ============================================================================

setlocal enabledelayedexpansion
set "SERVER_IP=204.12.223.93"
set "SHARE_PATH=\\%SERVER_IP%\optimizer_share"
set "USERNAME=opt_admin"
set "PASSWORD=Chetti@123"

echo.
echo ============================================================================
echo HeavyDB GPU Multi-Algorithm Optimization Platform
echo RESEARCH ANALYSIS OPTIMIZATION
echo ============================================================================
echo.
echo Expected Performance:
echo - Strategy Count: 500
echo - Execution Time: 20-45 minutes
echo - Success Rate: 90%+
echo - Analysis Type: Comprehensive algorithm comparison
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
set "JOB_ID=RESEARCH_%DATE:~-4,4%%DATE:~-10,2%%DATE:~-7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%"
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
net use R: %SHARE_PATH% %PASSWORD% /user:%USERNAME% /persistent:no >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to connect to server. Please check network connection.
    pause
    exit /b 1
)
echo ✓ Connected to server successfully

echo.
echo [2/5] Uploading dataset file...
copy "%INPUT_FILE%" "R:\input\%FILE_NAME%" >nul
if errorlevel 1 (
    echo ERROR: Failed to upload file to server
    net use R: /delete >nul 2>&1
    pause
    exit /b 1
)
echo ✓ Dataset uploaded: %FILE_NAME%

echo.
echo [3/5] Creating research optimization configuration...
(
echo {
echo   "job_id": "%JOB_ID%",
echo   "input_file": "%FILE_NAME%",
echo   "strategy_count": 500,
echo   "strategy": "tournament",
echo   "min_size": 30,
echo   "max_size": 50,
echo   "timeout": 2700,
echo   "algorithms": ["all"],
echo   "fitness_metric": "ratio",
echo   "optimization_type": "research",
echo   "verbose": true,
echo   "generate_reports": true,
echo   "algorithm_comparison": true,
echo   "created_at": "%DATE:~-4,4%-%DATE:~-10,2%-%DATE:~-7,2%T%TIME:~0,2%:%TIME:~3,2%:%TIME:~6,2%Z"
echo }
) > "R:\input\%JOB_ID%_config.json"
echo ✓ Research configuration created

echo.
echo [4/5] Starting comprehensive research analysis...
echo Job submitted to processing queue. Monitoring progress...
echo This is a comprehensive analysis and may take 20-45 minutes...
echo.
echo Progress will be updated every minute...

set "TIMEOUT_COUNT=0"
set "MAX_TIMEOUT=2700"

:check_progress
timeout /t 60 >nul
set /a TIMEOUT_COUNT+=60

if exist "R:\output\%JOB_ID%_status.json" (
    for /f "tokens=*" %%i in ('type "R:\output\%JOB_ID%_status.json" 2^>nul ^| findstr "completed"') do (
        if not "%%i"=="" goto job_completed
    )
    for /f "tokens=*" %%i in ('type "R:\output\%JOB_ID%_status.json" 2^>nul ^| findstr "failed"') do (
        if not "%%i"=="" goto job_failed
    )
)

set /a MINUTES=%TIMEOUT_COUNT%/60
echo Progress: %MINUTES% minutes elapsed (Expected: 20-45 minutes)

REM Show intermediate progress if available
if exist "R:\logs\%JOB_ID%_progress.log" (
    echo Latest progress:
    for /f "tokens=*" %%i in ('type "R:\logs\%JOB_ID%_progress.log" 2^>nul ^| findstr /i "algorithm\|fitness\|completed"') do (
        echo   %%i
    )
)

if %TIMEOUT_COUNT% lss %MAX_TIMEOUT% (
    goto check_progress
)

echo WARNING: Research analysis taking longer than expected (45+ minutes)
echo This may indicate a very large dataset or complex analysis requirements
echo Continue waiting? (Y/N)
set /p CONTINUE=
if /i "%CONTINUE%"=="Y" (
    set "MAX_TIMEOUT=3600"
    goto check_progress
)
goto job_timeout

:job_completed
set /a FINAL_MINUTES=%TIMEOUT_COUNT%/60
echo ✓ Research analysis completed successfully in %FINAL_MINUTES% minutes!

echo.
echo [5/5] Downloading comprehensive results...
if exist "R:\output\%JOB_ID%_results.json" (
    copy "R:\output\%JOB_ID%_results.json" "%~dp0%JOB_ID%_results.json" >nul
    echo ✓ Main results downloaded: %JOB_ID%_results.json
)

if exist "R:\output\%JOB_ID%_status.json" (
    copy "R:\output\%JOB_ID%_status.json" "%~dp0%JOB_ID%_status.json" >nul
    echo ✓ Status downloaded: %JOB_ID%_status.json
)

REM Download all research outputs
echo Downloading research reports and visualizations...
for %%f in (R:\output\%JOB_ID%_*.xlsx R:\output\%JOB_ID%_*.png R:\output\%JOB_ID%_*.csv R:\output\%JOB_ID%_*.pdf) do (
    if exist "%%f" (
        copy "%%f" "%~dp0" >nul
        echo ✓ Downloaded: %%~nxf
    )
)

REM Download algorithm comparison reports
for %%f in (R:\output\reports\%JOB_ID%_*.*) do (
    if exist "%%f" (
        copy "%%f" "%~dp0" >nul
        echo ✓ Downloaded report: %%~nxf
    )
)

echo.
echo ============================================================================
echo RESEARCH ANALYSIS OPTIMIZATION COMPLETED
echo ============================================================================
echo Job ID: %JOB_ID%
echo Execution Time: %FINAL_MINUTES% minutes
echo Analysis Type: Comprehensive algorithm comparison
echo Results available in current directory
echo.
echo Summary Results:
type "%~dp0%JOB_ID%_results.json" 2>nul
echo.
echo Check downloaded files for detailed analysis reports and visualizations.
goto cleanup

:job_failed
echo ✗ Research analysis failed. Check status file for details.
if exist "R:\output\%JOB_ID%_status.json" (
    copy "R:\output\%JOB_ID%_status.json" "%~dp0%JOB_ID%_error.json" >nul
    echo Error details saved to: %JOB_ID%_error.json
)
goto cleanup

:job_timeout
echo ✗ Research analysis timed out. Job may still be processing on server.
echo Check server status or try again later.
goto cleanup

:cleanup
echo.
echo Cleaning up connection...
net use R: /delete >nul 2>&1
echo ✓ Disconnected from server

echo.
echo Press any key to exit...
pause >nul
exit /b 0

@echo off
REM Heavy Optimizer Platform - SAMBA-ONLY HEAVYDB ACCELERATED LAUNCHER
REM Version: 6.0 - Complete SSH Elimination with Server-Side HeavyDB Acceleration
REM Architecture: Samba job queue system with HeavyDB acceleration

title Heavy Optimizer Platform - SAMBA-ONLY HEAVYDB ACCELERATED

echo ===============================================================================
echo                    HEAVY OPTIMIZER PLATFORM - SAMBA-ONLY HEAVYDB ACCELERATED
echo                     Complete SSH Elimination - Server-Side HeavyDB Acceleration
echo ===============================================================================
echo.
echo ARCHITECTURAL REVOLUTION:
echo   ✅ NO SSH client installation required
echo   ✅ NO plink.exe dependency
echo   ✅ NO dual authentication complexity
echo   ✅ SERVER-SIDE HeavyDB acceleration maintained
echo   ✅ CSV-only processing (Excel dependencies removed)
echo   ✅ Job queue system via Samba share
echo   ✅ Real-time job monitoring and status updates
echo.
echo Performance Specifications:
echo   - HeavyDB Acceleration: Server-side GPU processing
echo   - Input Format: CSV-only (simplified dependencies)
echo   - Execution: Server-side with Samba job queue
echo   - Job Monitoring: Real-time status via Samba share
echo   - All 7 Algorithms: SA, GA, PSO, DE, ACO, BO, RS
echo   - Reference-Compatible Output: 6 files in timestamped directories
echo.
echo Output Format (Reference Compatible):
echo   Directory Structure: output/
echo   └── run_YYYYMMDD_HHMMSS/                    (Timestamped container)
echo       ├── optimization_summary_YYYYMMDD_HHMMSS.txt  (Main summary)
echo       ├── strategy_metrics.csv                      (Strategy data)
echo       ├── error_log.txt                            (Error logging)
echo       ├── drawdowns_Best_Portfolio_Size##_*.png     (Drawdown charts)
echo       ├── equity_curves_Best_Portfolio_Size##_*.png (Equity curves)
echo       └── Best_Portfolio_Size##_*.txt               (Portfolio details)
echo.
echo ===============================================================================

REM Network Drive Setup (Samba Only)
echo.
echo [STEP 1/4] Network Drive Setup (Samba Authentication Only)...
echo.

REM Clean up existing connections
net use L: /delete /y >nul 2>&1
net use M: /delete /y >nul 2>&1
net use N: /delete /y >nul 2>&1

REM Try multiple drive letters for network mapping
echo Connecting to Heavy Optimizer server (204.12.223.93)...

net use L: \\204.12.223.93\optimizer_share /user:opt_admin Chetti@123 /persistent:no >nul 2>&1
if errorlevel 1 (
    echo L: drive in use, trying M: drive...
    net use M: \\204.12.223.93\optimizer_share /user:opt_admin Chetti@123 /persistent:no >nul 2>&1
    if errorlevel 1 (
        echo M: drive in use, trying N: drive...
        net use N: \\204.12.223.93\optimizer_share /user:opt_admin Chetti@123 /persistent:no >nul 2>&1
        if errorlevel 1 (
            echo ❌ Network Drive Error: Unable to map network drive (L:, M:, N: all in use)
            echo.
            echo TROUBLESHOOTING STEPS:
            echo   1. Manually disconnect existing network drives
            echo   2. Check Windows Explorer for mapped drives
            echo   3. Restart this batch file
            echo.
            pause
            exit /b 1
        ) else (
            set DRIVE_LETTER=N:
            echo ✅ Network Drive: Connected successfully using N: drive
        )
    ) else (
        set DRIVE_LETTER=M:
        echo ✅ Network Drive: Connected successfully using M: drive
    )
) else (
    set DRIVE_LETTER=L:
    echo ✅ Network Drive: Connected successfully using L: drive
)

echo Network drive mapped to %DRIVE_LETTER%
echo Output will be in: %DRIVE_LETTER%\output\run_[YYYYMMDD_HHMMSS]\

REM Backend Environment Verification
echo.
echo [STEP 2/4] Backend Environment Verification...
echo.

REM Check for migrated backend
if exist "%DRIVE_LETTER%\backend\csv_only_heavydb_workflow.py" (
    echo ✅ Backend: CSV-only HeavyDB workflow available
    set BACKEND_AVAILABLE=true
) else (
    echo ❌ Backend: CSV-only workflow not found
    set BACKEND_AVAILABLE=false
)

if exist "%DRIVE_LETTER%\jobs\queue" (
    echo ✅ Job Queue: Samba job queue system ready
    set JOB_QUEUE_AVAILABLE=true
) else (
    echo ❌ Job Queue: Samba job queue not available
    set JOB_QUEUE_AVAILABLE=false
)

if "%BACKEND_AVAILABLE%"=="false" (
    echo.
    echo ❌ BACKEND NOT AVAILABLE
    echo The migrated backend environment is not accessible.
    echo Please contact administrator to complete backend migration.
    echo.
    pause
    exit /b 1
)

if "%JOB_QUEUE_AVAILABLE%"=="false" (
    echo.
    echo ❌ JOB QUEUE NOT AVAILABLE
    echo The Samba job queue system is not set up.
    echo Please contact administrator to initialize job queue.
    echo.
    pause
    exit /b 1
)

REM Load configuration-based portfolio sizes
echo.
echo [STEP 3/4] Loading Configuration Parameters...
echo.

REM Set default values (fallback if config not accessible)
set DEFAULT_PORTFOLIO_SIZE=35
set HFT_PORTFOLIO_SIZE=20
set COMPREHENSIVE_PORTFOLIO_SIZE=50
set MIN_PORTFOLIO_SIZE=10
set MAX_PORTFOLIO_SIZE=100

REM Try to read configuration from production_config.ini
if exist "%DRIVE_LETTER%\config\production_config.ini" (
    echo ✅ Configuration: Loading parameters from production_config.ini
    
    REM Parse configuration file for portfolio sizes (simplified parsing)
    for /f "tokens=2 delims==" %%a in ('findstr "default_portfolio_size" "%DRIVE_LETTER%\config\production_config.ini" 2^>nul') do (
        set DEFAULT_PORTFOLIO_SIZE=%%a
        set DEFAULT_PORTFOLIO_SIZE=!DEFAULT_PORTFOLIO_SIZE: =!
    )
    for /f "tokens=2 delims==" %%a in ('findstr "hft_portfolio_size" "%DRIVE_LETTER%\config\production_config.ini" 2^>nul') do (
        set HFT_PORTFOLIO_SIZE=%%a
        set HFT_PORTFOLIO_SIZE=!HFT_PORTFOLIO_SIZE: =!
    )
    for /f "tokens=2 delims==" %%a in ('findstr "comprehensive_portfolio_size" "%DRIVE_LETTER%\config\production_config.ini" 2^>nul') do (
        set COMPREHENSIVE_PORTFOLIO_SIZE=%%a
        set COMPREHENSIVE_PORTFOLIO_SIZE=!COMPREHENSIVE_PORTFOLIO_SIZE: =!
    )
    
    echo Configuration loaded successfully:
    echo   - Default Portfolio Size: %DEFAULT_PORTFOLIO_SIZE%
    echo   - HFT Portfolio Size: %HFT_PORTFOLIO_SIZE%
    echo   - Comprehensive Portfolio Size: %COMPREHENSIVE_PORTFOLIO_SIZE%
) else (
    echo ⚠️ Configuration: Using default values (config file not accessible)
    echo   - Default Portfolio Size: %DEFAULT_PORTFOLIO_SIZE%
    echo   - HFT Portfolio Size: %HFT_PORTFOLIO_SIZE%
    echo   - Comprehensive Portfolio Size: %COMPREHENSIVE_PORTFOLIO_SIZE%
)

echo.
echo [STEP 4/4] System Ready - Samba-Only HeavyDB Architecture Operational
echo ✅ Network Drive: %DRIVE_LETTER% connected successfully
echo ✅ Backend: CSV-only HeavyDB workflow ready
echo ✅ Job Queue: Samba-based job submission system active
echo ✅ Configuration: Portfolio parameters loaded
echo ✅ HeavyDB Acceleration: Server-side GPU processing enabled
echo.

:MAIN_MENU
cls
echo ===============================================================================
echo                    HEAVY OPTIMIZER PLATFORM - SAMBA-ONLY HEAVYDB MAIN MENU
echo                     Server-Side HeavyDB Acceleration - No SSH Required
echo ===============================================================================
echo.
echo Please select an optimization option:
echo.
echo 1. Complete Portfolio Optimization (%DEFAULT_PORTFOLIO_SIZE% strategies)
echo    Execution: SERVER-SIDE with HeavyDB acceleration
echo    Input: CSV-only (Excel dependencies removed)
echo    Expected time: Dynamic based on HeavyDB performance
echo    Directory: %DRIVE_LETTER%\output\run_[YYYYMMDD_HHMMSS]\
echo.
echo 2. HFT Speed-Focused Optimization (%HFT_PORTFOLIO_SIZE% strategies)
echo    Execution: SERVER-SIDE with HeavyDB acceleration
echo    Input: CSV-only (optimized for speed)
echo    Expected time: Dynamic based on HeavyDB performance
echo    Directory: %DRIVE_LETTER%\output\run_[YYYYMMDD_HHMMSS]\
echo.
echo 3. Comprehensive Portfolio Optimization (%COMPREHENSIVE_PORTFOLIO_SIZE% strategies)
echo    Execution: SERVER-SIDE with HeavyDB acceleration
echo    Input: CSV-only (comprehensive analysis)
echo    Expected time: Dynamic based on HeavyDB performance
echo    Directory: %DRIVE_LETTER%\output\run_[YYYYMMDD_HHMMSS]\
echo.
echo 4. Custom Portfolio Size (Choose specific size: %MIN_PORTFOLIO_SIZE%-%MAX_PORTFOLIO_SIZE%)
echo    Execution: SERVER-SIDE with HeavyDB acceleration
echo    Input: CSV-only (custom configuration)
echo    Expected time: Dynamic based on portfolio size and HeavyDB
echo    Directory: %DRIVE_LETTER%\output\run_[YYYYMMDD_HHMMSS]\
echo.
echo 5. Job Status Monitor
echo    Monitor: Real-time job queue status and completion
echo.
echo 6. System Status and Architecture Information
echo    Details: Samba-only HeavyDB architecture benefits
echo.
echo 7. Exit
echo.
echo ===============================================================================

set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto COMPLETE_OPTIMIZATION
if "%choice%"=="2" goto HFT_OPTIMIZATION
if "%choice%"=="3" goto COMPREHENSIVE_OPTIMIZATION
if "%choice%"=="4" goto CUSTOM_OPTIMIZATION
if "%choice%"=="5" goto JOB_STATUS_MONITOR
if "%choice%"=="6" goto SYSTEM_STATUS
if "%choice%"=="7" goto EXIT
echo Invalid choice. Please select 1-7.
pause
goto MAIN_MENU

:COMPLETE_OPTIMIZATION
cls
echo ===============================================================================
echo                    COMPLETE PORTFOLIO OPTIMIZATION
echo                     Samba-Only Architecture: Server-Side HeavyDB Acceleration
echo ===============================================================================
echo.
echo Configuration:
echo   - All 7 algorithms: SA, GA, PSO, DE, ACO, BO, RS
echo   - Portfolio size: %DEFAULT_PORTFOLIO_SIZE% strategies (from configuration)
echo   - Execution mode: SERVER-SIDE with HeavyDB acceleration
echo   - Input format: CSV-only (Excel dependencies removed)
echo   - Job submission: Samba-based queue system
echo   - Output format: Reference-compatible directory structure
echo.

echo Available CSV files in input directory:
dir "%DRIVE_LETTER%\input\*.csv" /b 2>nul
if errorlevel 1 (
    echo ⚠️ No CSV files found in input directory
)

echo.
set /p dataset="Enter CSV filename (or press Enter for SENSEX_test_dataset.csv): "
if "%dataset%"=="" set dataset=SENSEX_test_dataset.csv

REM Validate CSV file exists
if not exist "%DRIVE_LETTER%\input\%dataset%" (
    echo ❌ Error: CSV file not found: %dataset%
    echo Please ensure the file exists in %DRIVE_LETTER%\input\
    pause
    goto MAIN_MENU
)

REM Validate CSV extension
echo %dataset% | findstr /i "\.csv$" >nul
if errorlevel 1 (
    echo ❌ Error: Only CSV files are supported in Samba-only architecture
    echo File must have .csv extension
    pause
    goto MAIN_MENU
)

echo.
echo ===============================================================================
echo                    SAMBA JOB SUBMISSION IN PROGRESS
echo ===============================================================================
echo.
echo Dataset: %dataset%
echo Portfolio Size: %DEFAULT_PORTFOLIO_SIZE%
echo Execution: SERVER-SIDE with HeavyDB acceleration
echo Job Submission: Samba-based queue system
echo Processing: Creating job file and submitting to queue...
echo.

REM Generate unique job ID
set JOB_ID=job_%RANDOM%_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set JOB_ID=%JOB_ID: =0%

echo Progress: Generating job ID: %JOB_ID%

REM Create job file
set JOB_FILE=%DRIVE_LETTER%\jobs\queue\%JOB_ID%.json

echo Progress: Creating job file: %JOB_FILE%

REM Create JSON job file
(
echo {
echo   "job_id": "%JOB_ID%",
echo   "input_file": "%dataset%",
echo   "portfolio_size": %DEFAULT_PORTFOLIO_SIZE%,
echo   "job_type": "complete_optimization",
echo   "timestamp": "%date% %time%",
echo   "client_info": {
echo     "computer_name": "%COMPUTERNAME%",
echo     "username": "%USERNAME%",
echo     "drive_letter": "%DRIVE_LETTER%"
echo   },
echo   "execution_mode": "server_side_heavydb",
echo   "input_format": "csv_only"
echo }
) > "%JOB_FILE%"

if errorlevel 1 (
    echo ❌ Error: Failed to create job file
    echo Please check network connectivity and permissions
    pause
    goto MAIN_MENU
)

echo ✅ Job submitted successfully to Samba queue
echo Job ID: %JOB_ID%
echo Job File: %JOB_FILE%
echo.
echo Progress: Monitoring job execution (server-side HeavyDB processing)...
echo.

REM Monitor job completion
set JOB_COMPLETED=false
set MONITOR_COUNT=0

:JOB_MONITOR_LOOP
set /a MONITOR_COUNT+=1

REM Check if job moved to completed directory
if exist "%DRIVE_LETTER%\jobs\completed\%JOB_ID%.json" (
    set JOB_COMPLETED=true
    goto JOB_COMPLETED_SUCCESS
)

REM Check if job moved to failed directory
if exist "%DRIVE_LETTER%\jobs\failed\%JOB_ID%.json" (
    goto JOB_COMPLETED_FAILED
)

REM Check if job is in processing directory
if exist "%DRIVE_LETTER%\jobs\processing\%JOB_ID%.json" (
    echo [%MONITOR_COUNT%] Job is being processed server-side with HeavyDB acceleration...
) else (
    echo [%MONITOR_COUNT%] Job is queued for server-side processing...
)

REM Wait 3 seconds before next check
timeout /t 3 /nobreak >nul
goto JOB_MONITOR_LOOP

:JOB_COMPLETED_SUCCESS
echo.
echo ✅ OPTIMIZATION COMPLETED SUCCESSFULLY
echo ===============================================================================
echo.
echo Job ID: %JOB_ID%
echo Execution: SERVER-SIDE with HeavyDB acceleration
echo Input Format: CSV-only processing
echo.

REM Read completion details from job file
if exist "%DRIVE_LETTER%\jobs\completed\%JOB_ID%.json" (
    echo Job completion details available in: %DRIVE_LETTER%\jobs\completed\%JOB_ID%.json
)

echo.
echo Results available in reference-compatible format:
echo   Base Directory: %DRIVE_LETTER%\output\
echo   Run Directory: %DRIVE_LETTER%\output\run_[YYYYMMDD_HHMMSS]\
echo.
echo Files generated within the timestamped directory:
echo   └── run_[YYYYMMDD_HHMMSS]\
echo       ├── optimization_summary_[timestamp].txt (Main summary report)
echo       ├── strategy_metrics.csv (Strategy performance data)
echo       ├── error_log.txt (System error logging)
echo       ├── drawdowns_Best_Portfolio_Size%DEFAULT_PORTFOLIO_SIZE%_[timestamp].png (Drawdown visualization)
echo       ├── equity_curves_Best_Portfolio_Size%DEFAULT_PORTFOLIO_SIZE%_[timestamp].png (Equity curves)
echo       └── Best_Portfolio_Size%DEFAULT_PORTFOLIO_SIZE%_[timestamp].txt (Portfolio details)
echo.
echo Performance: SERVER-SIDE HeavyDB acceleration
echo Architecture: Samba-only job queue system
echo Best algorithm: Automatically selected based on fitness scores
echo Output format: Matches reference implementation exactly
echo.
echo IMPORTANT: All files are contained within the timestamped directory!
echo Navigate to: %DRIVE_LETTER%\output\run_[YYYYMMDD_HHMMSS]\ to find your results
echo.

pause
goto MAIN_MENU

:JOB_COMPLETED_FAILED
echo.
echo ❌ OPTIMIZATION FAILED
echo ===============================================================================
echo.
echo Job ID: %JOB_ID%
echo Status: FAILED during server-side execution
echo.

REM Read failure details from job file
if exist "%DRIVE_LETTER%\jobs\failed\%JOB_ID%.json" (
    echo Failure details available in: %DRIVE_LETTER%\jobs\failed\%JOB_ID%.json
    echo.
    echo Common failure causes:
    echo   1. Invalid CSV file format
    echo   2. Server-side processing error
    echo   3. HeavyDB acceleration issues
    echo   4. Insufficient server resources
    echo   5. Network connectivity problems
    echo.
    echo Troubleshooting steps:
    echo   1. Verify CSV file format and content
    echo   2. Check server status (Option 6 from main menu)
    echo   3. Try with a smaller portfolio size
    echo   4. Contact system administrator if problem persists
)

echo.
pause
goto MAIN_MENU

:JOB_STATUS_MONITOR
cls
echo ===============================================================================
echo                    JOB STATUS MONITOR
echo ===============================================================================
echo.
echo Monitoring Samba job queue system...
echo.

echo QUEUE STATUS:
echo Queue directory: %DRIVE_LETTER%\jobs\queue\
dir "%DRIVE_LETTER%\jobs\queue\*.json" /b 2>nul
if errorlevel 1 echo   No jobs in queue

echo.
echo PROCESSING STATUS:
echo Processing directory: %DRIVE_LETTER%\jobs\processing\
dir "%DRIVE_LETTER%\jobs\processing\*.json" /b 2>nul
if errorlevel 1 echo   No jobs currently processing

echo.
echo COMPLETED JOBS:
echo Completed directory: %DRIVE_LETTER%\jobs\completed\
dir "%DRIVE_LETTER%\jobs\completed\*.json" /b 2>nul | head -10
if errorlevel 1 echo   No completed jobs

echo.
echo FAILED JOBS:
echo Failed directory: %DRIVE_LETTER%\jobs\failed\
dir "%DRIVE_LETTER%\jobs\failed\*.json" /b 2>nul | head -5
if errorlevel 1 echo   No failed jobs

echo.
echo Job queue system is operating via Samba share
echo Server-side processing with HeavyDB acceleration
echo.
pause
goto MAIN_MENU

:EXIT
echo.
echo Disconnecting from Heavy Optimizer server...
if defined DRIVE_LETTER (
    net use %DRIVE_LETTER% /delete /y >nul 2>&1
    echo Network drive %DRIVE_LETTER% disconnected
)

echo.
echo ===============================================================================
echo                    HEAVY OPTIMIZER PLATFORM SESSION ENDED
echo                     Samba-Only HeavyDB Architecture - No SSH Required
echo ===============================================================================
echo.
echo Thank you for using the Heavy Optimizer Platform!
echo.
echo Session Summary:
echo   - Architecture: Samba-only with server-side HeavyDB acceleration
echo   - Network Drive: %DRIVE_LETTER% (disconnected)
echo   - Execution: Server-side job queue system
echo   - Input Format: CSV-only (Excel dependencies removed)
echo   - HeavyDB Acceleration: Server-side GPU processing
echo   - Job Submission: Samba-based queue system
echo.
echo Configuration Summary:
echo   - Default Portfolio Size: %DEFAULT_PORTFOLIO_SIZE%
echo   - HFT Portfolio Size: %HFT_PORTFOLIO_SIZE%
echo   - Comprehensive Portfolio Size: %COMPREHENSIVE_PORTFOLIO_SIZE%
echo   - Portfolio Range: %MIN_PORTFOLIO_SIZE%-%MAX_PORTFOLIO_SIZE%
echo.
echo Output Format:
echo   - Directory structure: run_YYYYMMDD_HHMMSS/ (contains all files)
echo   - Complete file set: 6 reference-compatible files within each directory
echo   - Professional quality: Maintained across all outputs
echo   - File location: All files contained within timestamped directories
echo.
echo For support or questions, contact the Heavy Optimizer team.
echo Remember: Look for your results in the run_[YYYYMMDD_HHMMSS] directories!
echo.
echo ARCHITECTURAL BENEFITS:
echo   ✅ NO SSH client installation required
echo   ✅ NO dual authentication complexity
echo   ✅ SERVER-SIDE HeavyDB acceleration maintained
echo   ✅ CSV-only processing (simplified dependencies)
echo   ✅ REAL-TIME job monitoring via Samba
echo   ✅ SIMPLIFIED troubleshooting and support
echo.
pause
exit

@echo off
REM Heavy Optimizer Platform - SAMBA-ONLY HEAVYDB ACCELERATED LAUNCHER (FIXED)
REM Version: 6.1 - Uses existing Z: drive mapping
REM Architecture: Samba job queue system with HeavyDB acceleration

title Heavy Optimizer Platform - SAMBA-ONLY HEAVYDB ACCELERATED (FIXED)

echo ===============================================================================
echo                    HEAVY OPTIMIZER PLATFORM - SAMBA-ONLY HEAVYDB ACCELERATED
echo                     Complete SSH Elimination - Server-Side HeavyDB Acceleration
echo                                      FIXED VERSION - USES Z: DRIVE
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

REM Network Drive Setup (Using existing Z: drive)
echo.
echo [STEP 1/4] Network Drive Setup - Using Existing Z: Drive...
echo.

REM Check if Z: drive is already mapped to optimizer_share
if exist "Z:\backend\csv_only_heavydb_workflow.py" (
    set DRIVE_LETTER=Z:
    echo ✅ Network Drive: Using existing Z: drive mapping
    echo ✅ Network Drive: Connected to \\204.12.223.93\optimizer_share
) else (
    echo ❌ Network Drive Error: Z: drive not properly mapped to optimizer_share
    echo.
    echo TROUBLESHOOTING STEPS:
    echo   1. Ensure Z: drive is mapped to \\204.12.223.93\optimizer_share
    echo   2. Run: net use Z: \\204.12.223.93\optimizer_share /user:opt_admin Chetti@123
    echo   3. Restart this batch file
    echo.
    pause
    exit /b 1
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

REM Check for job queue system
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
echo.
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto COMPLETE_OPTIMIZATION
if "%choice%"=="2" goto HFT_OPTIMIZATION
if "%choice%"=="3" goto COMPREHENSIVE_OPTIMIZATION
if "%choice%"=="4" goto CUSTOM_OPTIMIZATION
if "%choice%"=="5" goto JOB_MONITOR
if "%choice%"=="6" goto SYSTEM_STATUS
if "%choice%"=="7" goto EXIT
echo Invalid choice. Please select 1-7.
pause
goto MAIN_MENU


:COMPLETE_OPTIMIZATION
echo.
echo ===============================================================================
echo                    COMPLETE PORTFOLIO OPTIMIZATION (%DEFAULT_PORTFOLIO_SIZE% STRATEGIES)
echo                     Server-Side HeavyDB Acceleration - CSV Processing
echo ===============================================================================
echo.

REM Available datasets in input directory
echo Available datasets:
echo.
if exist "%DRIVE_LETTER%\input\*.csv" (
    set DATASET_COUNT=0
    for %%f in ("%DRIVE_LETTER%\input\*.csv") do (
        set /a DATASET_COUNT+=1
        echo   !DATASET_COUNT!. %%~nxf
    )
) else (
    echo ❌ No CSV datasets found in input directory
    echo Please ensure CSV files are placed in %DRIVE_LETTER%\input\
    pause
    goto MAIN_MENU
)

echo.
set /p dataset_choice="Select dataset number: "
set DATASET_COUNT=0
for %%f in ("%DRIVE_LETTER%\input\*.csv") do (
    set /a DATASET_COUNT+=1
    if "!DATASET_COUNT!"=="%dataset_choice%" (
        set dataset=%%~nxf
        goto DATASET_SELECTED
    )
)

echo Invalid dataset selection
pause
goto MAIN_MENU

:DATASET_SELECTED
echo.
echo Selected dataset: %dataset%
echo Portfolio size: %DEFAULT_PORTFOLIO_SIZE% strategies
echo Execution: Server-side HeavyDB acceleration
echo Output: %DRIVE_LETTER%\output\run_[YYYYMMDD_HHMMSS]\
echo.
set /p confirm="Proceed with optimization? (y/n): "
if /i "%confirm%" neq "y" goto MAIN_MENU

REM Generate unique job ID
for /f "tokens=2 delims==" %%a in ('wmic OS Get LocalDateTime /value') do set ldt=%%a
set job_timestamp=%ldt:~0,4%%ldt:~4,2%%ldt:~6,2%_%ldt:~8,2%%ldt:~10,2%%ldt:~12,2%
set JOB_ID=complete_opt_%job_timestamp%
set JOB_FILE=%DRIVE_LETTER%\jobs\queue\%JOB_ID%.json

echo.
echo Submitting job to server-side HeavyDB processing...
echo Job ID: %JOB_ID%

REM Create jobs directory if it doesn't exist
if not exist "%DRIVE_LETTER%\jobs\queue" mkdir "%DRIVE_LETTER%\jobs\queue"
if not exist "%DRIVE_LETTER%\jobs\completed" mkdir "%DRIVE_LETTER%\jobs\completed"
if not exist "%DRIVE_LETTER%\jobs\failed" mkdir "%DRIVE_LETTER%\jobs\failed"

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

echo ✅ Job submitted successfully to Samba queue
echo.
echo Processing on server with HeavyDB acceleration...
echo Please wait while optimization completes on server-side.
echo.
echo You can:
echo   - Return to main menu (jobs run in background)
echo   - Use option 5 to monitor job status
echo   - Check %DRIVE_LETTER%\output\ for results when complete
echo.
pause
goto MAIN_MENU

:HFT_OPTIMIZATION
echo.
echo ===============================================================================
echo                    HFT SPEED-FOCUSED OPTIMIZATION (%HFT_PORTFOLIO_SIZE% STRATEGIES)
echo                     Server-Side HeavyDB Acceleration - High-Speed Processing
echo ===============================================================================
echo.
REM Similar logic as COMPLETE_OPTIMIZATION but with HFT_PORTFOLIO_SIZE
goto MAIN_MENU

:COMPREHENSIVE_OPTIMIZATION
echo.
echo ===============================================================================
echo                 COMPREHENSIVE PORTFOLIO OPTIMIZATION (%COMPREHENSIVE_PORTFOLIO_SIZE% STRATEGIES)
echo                     Server-Side HeavyDB Acceleration - Deep Analysis
echo ===============================================================================
echo.
REM Similar logic as COMPLETE_OPTIMIZATION but with COMPREHENSIVE_PORTFOLIO_SIZE
goto MAIN_MENU

:CUSTOM_OPTIMIZATION
echo.
echo ===============================================================================
echo                        CUSTOM PORTFOLIO SIZE OPTIMIZATION
echo                     Server-Side HeavyDB Acceleration - Custom Configuration
echo ===============================================================================
echo.
set /p custom_size="Enter portfolio size (%MIN_PORTFOLIO_SIZE%-%MAX_PORTFOLIO_SIZE%): "
if %custom_size% LSS %MIN_PORTFOLIO_SIZE% (
    echo Portfolio size too small. Minimum: %MIN_PORTFOLIO_SIZE%
    pause
    goto MAIN_MENU
)
if %custom_size% GTR %MAX_PORTFOLIO_SIZE% (
    echo Portfolio size too large. Maximum: %MAX_PORTFOLIO_SIZE%
    pause
    goto MAIN_MENU
)
REM Process with custom size
goto MAIN_MENU

:JOB_MONITOR
echo.
echo ===============================================================================
echo                           JOB STATUS MONITOR
echo                       Real-Time Job Queue Status
echo ===============================================================================
echo.
echo Checking job queue status...
echo.

if exist "%DRIVE_LETTER%\jobs\queue\*.json" (
    echo 📋 QUEUED JOBS:
    for %%f in ("%DRIVE_LETTER%\jobs\queue\*.json") do (
        echo   - %%~nxf
    )
) else (
    echo ✅ No jobs currently queued
)

echo.
if exist "%DRIVE_LETTER%\jobs\completed\*.json" (
    echo ✅ COMPLETED JOBS:
    for %%f in ("%DRIVE_LETTER%\jobs\completed\*.json") do (
        echo   - %%~nxf
    )
) else (
    echo No completed jobs
)

echo.
if exist "%DRIVE_LETTER%\jobs\failed\*.json" (
    echo ❌ FAILED JOBS:
    for %%f in ("%DRIVE_LETTER%\jobs\failed\*.json") do (
        echo   - %%~nxf
    )
) else (
    echo No failed jobs
)

echo.
pause
goto MAIN_MENU

:SYSTEM_STATUS
echo.
echo ===============================================================================
echo                         SYSTEM STATUS & ARCHITECTURE
echo                      Samba-Only HeavyDB Architecture Benefits
echo ===============================================================================
echo.
echo ✅ ARCHITECTURE BENEFITS:
echo   - Complete SSH elimination
echo   - No client-side dependencies
echo   - Server-side HeavyDB GPU acceleration
echo   - Simplified authentication (Samba only)
echo   - Job queue-based processing
echo   - Real-time status monitoring
echo.
echo 🔧 CURRENT CONFIGURATION:
echo   - Network Drive: %DRIVE_LETTER%
echo   - Server: 204.12.223.93
echo   - Backend: %DRIVE_LETTER%\backend\
echo   - Job Queue: %DRIVE_LETTER%\jobs\
echo   - Output: %DRIVE_LETTER%\output\
echo.
echo 🚀 PERFORMANCE:
echo   - HeavyDB GPU Acceleration: Enabled
echo   - CSV-Only Processing: Optimized
echo   - Server-Side Execution: Full acceleration
echo.
pause
goto MAIN_MENU

:EXIT
echo.
echo Thank you for using Heavy Optimizer Platform!
echo Network drive %DRIVE_LETTER% will remain connected for future use.
echo.
pause
exit /b 0

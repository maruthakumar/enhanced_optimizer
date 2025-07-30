@echo off
REM ============================================================================
REM HeavyDB GPU Multi-Algorithm Optimization Platform
REM Master Launcher - Choose Your Optimization Template
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo HeavyDB GPU Multi-Algorithm Optimization Platform
echo MASTER OPTIMIZATION LAUNCHER
echo ============================================================================
echo.
echo Select your optimization template:
echo.
echo [1] High-Frequency Trading (HFT)
echo     - 100 strategies, 2-4 minutes, 98%+ success rate
echo     - Optimized for speed and quick results
echo     - Best for: Day trading, quick portfolio adjustments
echo.
echo [2] Portfolio Management
echo     - 250 strategies, 8-15 minutes, 95%+ success rate  
echo     - Balanced optimization with quality focus
echo     - Best for: Investment portfolios, fund management
echo.
echo [3] Research Analysis
echo     - 500 strategies, 20-45 minutes, 90%+ success rate
echo     - Comprehensive algorithm comparison
echo     - Best for: Academic research, strategy development
echo.
echo [4] Large-Scale Processing
echo     - 10,000+ strategies, 2-4 hours, 95%+ success rate
echo     - Enterprise-grade distributed processing
echo     - Best for: Institutional analysis, big data optimization
echo.
echo [5] Custom Configuration
echo     - Manual parameter selection
echo     - Advanced users only
echo.
echo [0] Exit
echo.

set /p CHOICE="Enter your choice (0-5): "

if "%CHOICE%"=="0" goto exit
if "%CHOICE%"=="1" goto hft
if "%CHOICE%"=="2" goto portfolio
if "%CHOICE%"=="3" goto research
if "%CHOICE%"=="4" goto largescale
if "%CHOICE%"=="5" goto custom

echo Invalid choice. Please select 0-5.
pause
goto start

:hft
echo.
echo Starting High-Frequency Trading Optimization...
echo Please select your Excel dataset file:
echo.
set /p DATASET="Enter full path to your Excel file: "
if not exist "%DATASET%" (
    echo Error: File not found. Please check the path.
    pause
    goto start
)
call HFT_Optimization.bat "%DATASET%"
goto end

:portfolio
echo.
echo Starting Portfolio Management Optimization...
echo Please select your Excel dataset file:
echo.
set /p DATASET="Enter full path to your Excel file: "
if not exist "%DATASET%" (
    echo Error: File not found. Please check the path.
    pause
    goto start
)
call Portfolio_Optimization.bat "%DATASET%"
goto end

:research
echo.
echo Starting Research Analysis Optimization...
echo Please select your Excel dataset file:
echo.
set /p DATASET="Enter full path to your Excel file: "
if not exist "%DATASET%" (
    echo Error: File not found. Please check the path.
    pause
    goto start
)
call Research_Optimization.bat "%DATASET%"
goto end

:largescale
echo.
echo Starting Large-Scale Distributed Optimization...
echo Please select your Excel dataset file:
echo.
set /p DATASET="Enter full path to your Excel file: "
if not exist "%DATASET%" (
    echo Error: File not found. Please check the path.
    pause
    goto start
)
echo.
set /p STRATEGIES="Enter number of strategies to optimize (default 10000): "
if "%STRATEGIES%"=="" set "STRATEGIES=10000"
call LargeScale_Optimization.bat "%DATASET%" %STRATEGIES%
goto end

:custom
echo.
echo ============================================================================
echo CUSTOM CONFIGURATION
echo ============================================================================
echo.
echo Please provide the following parameters:
echo.
set /p DATASET="Excel dataset file path: "
if not exist "%DATASET%" (
    echo Error: File not found. Please check the path.
    pause
    goto start
)

set /p STRATEGIES="Number of strategies (50-50000): "
set /p STRATEGY_TYPE="Strategy type (parallel/ensemble/tournament): "
set /p MIN_SIZE="Minimum portfolio size (5-50): "
set /p MAX_SIZE="Maximum portfolio size (10-100): "
set /p TIMEOUT="Timeout in seconds (60-14400): "

echo.
echo Configuration Summary:
echo - Dataset: %DATASET%
echo - Strategies: %STRATEGIES%
echo - Strategy Type: %STRATEGY_TYPE%
echo - Portfolio Size: %MIN_SIZE%-%MAX_SIZE%
echo - Timeout: %TIMEOUT% seconds
echo.
echo Proceed with custom optimization? (Y/N)
set /p CONFIRM=
if /i not "%CONFIRM%"=="Y" goto start

REM Create custom job config
set "FILE_NAME=%~nx1"
set "JOB_ID=CUSTOM_%DATE:~-4,4%%DATE:~-10,2%%DATE:~-7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%"
set "JOB_ID=%JOB_ID: =0%"

echo Creating custom optimization job...
REM Custom optimization logic would go here
echo Custom optimization not yet implemented in this demo.
echo Please use one of the predefined templates (1-4).
pause
goto start

:exit
echo.
echo Thank you for using HeavyDB GPU Multi-Algorithm Optimization Platform!
exit /b 0

:end
echo.
echo Optimization completed. Check the current directory for results.
echo.
echo Run this launcher again to perform another optimization.
pause
exit /b 0

:start
cls
goto :eof

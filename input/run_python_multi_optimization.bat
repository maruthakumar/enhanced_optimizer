@echo off
echo Starting Heavy Optimizer with Python_Multi_Consolidated dataset...
echo Portfolio Size: 35 strategies
echo Dataset: Python_Multi_Consolidated_20250726_161921.csv
echo.

REM Try different Python interpreters
echo Trying python3...
python3 Z:\backend\csv_only_heavydb_workflow.py Z:\input\Python_Multi_Consolidated_20250726_161921.csv 35

if errorlevel 1 (
    echo.
    echo Trying python...
    python Z:\backend\csv_only_heavydb_workflow.py Z:\input\Python_Multi_Consolidated_20250726_161921.csv 35
)

if errorlevel 1 (
    echo.
    echo Trying py...
    py Z:\backend\csv_only_heavydb_workflow.py Z:\input\Python_Multi_Consolidated_20250726_161921.csv 35
)

echo.
echo Optimization complete. Check Z:\output\ for results.
pause
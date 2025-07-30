#!/usr/bin/env python
"""
Strategy Consolidator - Local Version
----------------------------------------
This script consolidates strategy data from various file formats as defined in config_consol.ini.
It processes input files from multiple sub-folders under the "Input" folder.

Supported formats include:
  • FORMAT_1_BACKINZO_CSV        (e.g. Backinzo_Files)
  • FORMAT_2_PYTHON_XLSX         (e.g. Python_Files)
  • FORMAT_3_TRADING_VIEW_CSV    (e.g. TV_Files)
  • FORMAT_4_CONSOLIDATED_XLSX   (e.g. Consolidated_Files)
  • FORMAT_5_BACKINZO_Multi_CSV  (e.g. Backinzo_Multi_Files)
  • FORMAT_6_PYTHON_MULTI_XLSX   (e.g. Python_Multi_Files)
  • FORMAT_7_TradingView_Zone    (e.g. TV_Zone_Files)
  • FORMAT_8_PYTHON_MULTI_ZONE_XLSX  (e.g. Python_Multi_Zone_Files)

For zone files (FORMAT_7 and FORMAT_8), specialized processing is applied to assign zones based on market times.
For other formats, generic processing is used to parse dates, rename the PNL column, and add strategy and day information.

The consolidated data (pivoted on merge keys) is saved in the Output/Optimize folder.
If the consolidated dataset exceeds 500 strategy columns or 50,000 rows, the output is split into separate Excel files.
Logging output appears both on the console and in a file in the "log" folder.
"""

import os
import sys
import glob
import pandas as pd
import configparser
import numpy as np
import logging
import datetime
import time
import requests
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from openpyxl import Workbook

# ------------------ Global Setup ------------------

BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, 'Input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')
CONFIG_PATH = os.path.join(BASE_DIR, 'config_consol.ini')
LOG_DIR = os.path.join(BASE_DIR, 'log')

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Optimize"), exist_ok=True)

# Create folders for all supported formats
for folder in ["Backinzo_Files", "Python_Files", "TV_Files", "Consolidated_Files",
               "Backinzo_Multi_Files", "Python_Multi_Files", "TV_Zone_Files", "Python_Multi_Zone_Files"]:
    os.makedirs(os.path.join(INPUT_DIR, folder), exist_ok=True)

# Configure logging
log_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"consolidator_log_{log_timestamp}.txt")

# Remove any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Log startup information
logging.info("="*50)
logging.info("Strategy Consolidator Starting")
logging.info(f"Log file: {log_file}")
logging.info(f"Input directory: {INPUT_DIR}")
logging.info(f"Output directory: {OUTPUT_DIR}")
logging.info("="*50)

DEBUG = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

# ------------------ Utility Functions ------------------

def compute_drawdown(series: pd.Series) -> float:
    equity = series.cumsum()
    running_max = equity.cummax()
    return (running_max - equity).max()

def parse_multiple_date_formats(series, date_formats, verbose=False):
    if verbose:
        debug_print("Trying date formats:", date_formats)
    for fmt in date_formats.split(','):
        fmt = fmt.strip()
        try:
            parsed = pd.to_datetime(series, format=fmt, errors='coerce')
            if parsed.notna().sum() > 0:
                if verbose:
                    debug_print(f"Format '{fmt}' succeeded.")
                return parsed
        except Exception as ex:
            if verbose:
                debug_print(f"Exception with format '{fmt}':", ex)
            continue
    try:
        parsed = pd.to_datetime(series, errors='coerce')
        if parsed.notna().sum() > 0:
            if verbose:
                debug_print("Auto-detection succeeded")
            return parsed
    except Exception as e:
        if verbose:
            debug_print("Auto-detection failed:", e)
    return None

def update_progress(percent=0, message="Ready to start"):
    progress_file = os.path.join(LOG_DIR, 'progress.txt')
    with open(progress_file, 'w') as f:
        f.write(f'{percent},{message}')

def check_should_stop():
    stop_file = os.path.join(LOG_DIR, 'stop_processing.flag')
    return os.path.exists(stop_file)

def send_telegram_notification(bot_token, chat_id, message, disable_on_error=False, max_retries=3, retry_delay=5):
    if not bot_token or not chat_id:
        logging.info("Telegram notification skipped - missing bot token or chat ID")
        return False
    for attempt in range(max_retries):
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
            response = requests.post(url, data=data, timeout=30)
            if response.status_code == 200:
                logging.info("Telegram notification sent successfully")
                return True
            else:
                logging.warning(f"Telegram API error (attempt {attempt+1}/{max_retries}): {response.text}")
                if disable_on_error:
                    return False
        except Exception as e:
            logging.warning(f"Telegram error (attempt {attempt+1}/{max_retries}): {str(e)}")
            if disable_on_error:
                return False
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    return False

# ------------------ Pre-Reformat Functions ------------------

def pre_reformat_backinzo_csv(file_path, date_formats_pre):
    debug_print(f"Pre-reformatting Backinzo CSV: {file_path}")
    try:
        df = pd.read_csv(file_path, skiprows=7, header=0, encoding="utf-8")
        debug_print("Columns before rename:", df.columns.tolist())
        time_cols = [c for c in df.columns if c.startswith("Time")]
        if len(time_cols) > 1:
            for col in time_cols[1:]:
                df.rename(columns={col: "Time.2"}, inplace=True)
        if "Date" not in df.columns:
            logging.warning(f"[PreReformat] 'Date' column not found in {file_path}.")
            return False
        parsed = parse_multiple_date_formats(df["Date"], date_formats_pre, verbose=DEBUG)
        if parsed is None:
            logging.warning(f"[PreReformat] Could not parse dates in {file_path}.")
            return False
        df["Date"] = parsed.dt.strftime('%d-%b-%Y')
        df.dropna(subset=["Date"], inplace=True)
        if df.empty:
            logging.warning(f"[PreReformat] All rows dropped in {file_path}.")
            return False
        df.to_csv(file_path, index=False, encoding="utf-8")
        return True
    except Exception as ex:
        logging.error(f"[PreReformat] Error in {file_path}: {ex}")
        return False

def pre_reformat_backinzo_multi_csv(file_path, date_formats_pre):
    debug_print(f"Pre-reformatting Backinzo Multi CSV: {file_path}")
    try:
        df = pd.read_csv(file_path, skiprows=7, header=0, encoding="utf-8")
        debug_print("Columns before rename:", df.columns.tolist())
        time_cols = [c for c in df.columns if c.startswith("Time")]
        if len(time_cols) > 1:
            for col in time_cols[1:]:
                df.rename(columns={col: "Time.2"}, inplace=True)
        if "Date" not in df.columns:
            logging.warning(f"[PreReformatMulti] 'Date' column not found in {file_path}.")
            return False
        parsed = parse_multiple_date_formats(df["Date"], date_formats_pre, verbose=DEBUG)
        if parsed is None:
            logging.warning(f"[PreReformatMulti] Could not parse dates in {file_path}.")
            return False
        df["Date"] = parsed.dt.strftime('%d-%b-%Y')
        df.dropna(subset=["Date"], inplace=True)
        if df.empty:
            logging.warning(f"[PreReformatMulti] All rows dropped in {file_path}.")
            return False
        df.to_csv(file_path, index=False, encoding="utf-8")
        return True
    except Exception as ex:
        logging.error(f"[PreReformatMulti] Error in {file_path}: {ex}")
        return False

# ------------------ Processing Functions for Each Format ------------------
# (Processing functions for FORMAT_1-FORMAT_7 remain unchanged.)

def process_backinzo_csv(file_path, date_col, date_fmt_main, pnl_col):
    logging.info(f"[Backinzo] Processing file: {file_path}")
    try:
        df = pd.read_csv(file_path, header=0, encoding="utf-8")
        df.columns = df.columns.str.strip()
        if date_col not in df.columns:
            logging.warning(f"[Backinzo] Date column '{date_col}' not found in {file_path}.")
            return None
        parsed_dates = parse_multiple_date_formats(df[date_col], date_fmt_main, verbose=DEBUG)
        if parsed_dates is None:
            logging.warning(f"[Backinzo] Failed to parse dates in {file_path}.")
            return None
        df[date_col] = parsed_dates.dt.strftime('%Y-%m-%d')
        df.dropna(subset=[date_col], inplace=True)
        if pnl_col not in df.columns:
            logging.warning(f"[Backinzo] PNL column '{pnl_col}' not found in {file_path}.")
            return None
        df.rename(columns={date_col: "Date", pnl_col: "PNL"}, inplace=True)
        df["Day"] = pd.to_datetime(df["Date"], errors="coerce").dt.day_name()
        strategy_name = os.path.splitext(os.path.basename(file_path))[0]
        df["Strategy"] = strategy_name
        df = df[["Strategy", "Date", "Day", "PNL"]]
        logging.info(f"[Backinzo] Processed {len(df)} rows from {file_path}.")
        return df
    except Exception as e:
        logging.error(f"[Backinzo] Error processing {file_path}: {e}")
        return None

def process_backinzo_multi_file(file_path, date_fmt_main):
    logging.info(f"[Backinzo_Multi] Processing file: {file_path}")
    try:
        df = pd.read_csv(file_path, header=0, encoding="utf-8")
        df.columns = df.columns.str.strip()
        if len(df.columns) < 2 or df.columns[1].strip() != "Date":
            logging.warning(f"[Backinzo_Multi] 'Date' column not found at expected position in {file_path}.")
            return None
        if len(df.columns) < 14:
            logging.warning(f"[Backinzo_Multi] Not enough columns for strategy names in {file_path}.")
            return None
        raw_strategy_names = list(df.columns[13:])
        file_prefix = os.path.splitext(os.path.basename(file_path))[0]
        new_strategy_names = [f"{file_prefix}_{col}" for col in raw_strategy_names]
        cols = list(df.columns)
        for i, new_name in enumerate(new_strategy_names, start=13):
            cols[i] = new_name
        df.columns = cols
        try:
            df["Date"] = pd.to_datetime(df["Date"], format=date_fmt_main, errors="coerce").dt.strftime('%Y-%m-%d')
        except Exception as e:
            logging.warning(f"[Backinzo_Multi] Error parsing 'Date' in {file_path}: {e}")
            return None
        df.dropna(subset=["Date"], inplace=True)
        if df.empty:
            logging.warning(f"[Backinzo_Multi] No valid dates in {file_path}.")
            return None
        df["Day"] = pd.to_datetime(df["Date"], errors="coerce").dt.day_name()
        selected_cols = ["Date", "Day"] + new_strategy_names
        df_result = df[selected_cols].copy()
        logging.info(f"[Backinzo_Multi] Processed {len(df_result)} rows from {file_path}.")
        return df_result
    except Exception as e:
        logging.error(f"[Backinzo_Multi] Error processing {file_path}: {e}")
        return None

def process_python_xlsx(file_path, date_col, date_fmt, pnl_col, sheet_name):
    logging.info(f"[Python] Processing file: {file_path}")
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name) if sheet_name else pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        if date_col not in df.columns:
            logging.warning(f"[Python] Date column '{date_col}' not found in {file_path}.")
            return None
        date_fmt = date_fmt.replace("%%", "%")
        df[date_col] = pd.to_datetime(df[date_col], format=date_fmt, errors="coerce")
        df.dropna(subset=[date_col], inplace=True)
        if df.empty:
            logging.warning(f"[Python] No valid dates in {file_path}.")
            return None
        df[date_col] = df[date_col].dt.strftime('%Y-%m-%d')
        if pnl_col not in df.columns:
            logging.warning(f"[Python] PNL column '{pnl_col}' not found in {file_path}.")
            return None
        df.rename(columns={date_col: "Date", pnl_col: "PNL"}, inplace=True)
        df["Day"] = pd.to_datetime(df["Date"], errors="coerce").dt.day_name()
        strategy_name = os.path.splitext(os.path.basename(file_path))[0]
        df["Strategy"] = strategy_name
        df = df[["Strategy", "Date", "Day", "PNL"]]
        logging.info(f"[Python] Processed {len(df)} rows from {file_path}.")
        return df
    except Exception as e:
        logging.error(f"[Python] Error processing {file_path}: {e}")
        return None

def process_python_multi_xlsx(file_path, date_col, date_fmt, pnl_col):
    logging.info(f"[Python_Multi] Processing file: {file_path}")
    try:
        excel_file = pd.ExcelFile(file_path)
        sheets = excel_file.sheet_names
        logging.info(f"[Python_Multi] Found sheets in {file_path}: {sheets}")
        multi_data = []
        
        for sheet in sheets:
            if sheet.strip().lower() == "portfolio trans".lower():
                logging.info(f"[Python_Multi] Skipping 'Portfolio Trans' sheet")
                continue
                
            if sheet.strip().endswith("Trans"):
                strategy_name = sheet.strip()
                if strategy_name.lower().endswith("trans"):
                    strategy_name = strategy_name[:-5].strip()
                    
                logging.info(f"[Python_Multi] Processing sheet '{sheet}' as strategy '{strategy_name}'")
                df = pd.read_excel(file_path, sheet_name=sheet)
                
                # Clean column names
                df.columns = df.columns.str.strip()
                logging.info(f"[Python_Multi] Columns in sheet '{sheet}': {', '.join(df.columns)}")
                
                # Check for date column
                if date_col not in df.columns:
                    logging.warning(f"[Python_Multi] Date column '{date_col}' not found in sheet '{sheet}'. Available columns: {', '.join(df.columns)}")
                    continue
                
                # Process date column
                date_fmt = date_fmt.replace("%%", "%")
                df[date_col] = pd.to_datetime(df[date_col], format=date_fmt, errors="coerce")
                df.dropna(subset=[date_col], inplace=True)
                if df.empty:
                    logging.warning(f"[Python_Multi] No valid dates in sheet '{sheet}'.")
                    continue
                df[date_col] = df[date_col].dt.strftime('%Y-%m-%d')
                
                # Check for PNL column - strict matching
                if pnl_col not in df.columns:
                    logging.warning(f"[Python_Multi] PNL column '{pnl_col}' not found in sheet '{sheet}'. Available columns: {', '.join(df.columns)}")
                    continue
                
                # Create a new DataFrame with only the required columns
                result_df = pd.DataFrame()
                result_df["Date"] = df[date_col]
                result_df["PNL"] = df[pnl_col]
                result_df["Day"] = pd.to_datetime(df[date_col], errors="coerce").dt.day_name()
                result_df["Strategy"] = strategy_name
                
                # Ensure we only have the columns we want
                result_df = result_df[["Strategy", "Date", "Day", "PNL"]]
                
                # Drop any rows with NaN values
                result_df = result_df.dropna()
                
                if not result_df.empty:
                    logging.info(f"[Python_Multi] Successfully processed sheet '{sheet}' with {len(result_df)} rows")
                    multi_data.append(result_df)
                else:
                    logging.warning(f"[Python_Multi] No valid data after processing sheet '{sheet}'")
        
        if not multi_data:
            logging.warning(f"[Python_Multi] No qualifying sheets found in {file_path}.")
            return None
        
        combined_df = pd.concat(multi_data, ignore_index=True)
        logging.info(f"[Python_Multi] Processed {len(combined_df)} total rows from {file_path}.")
        return combined_df
        
    except Exception as e:
        logging.error(f"[Python_Multi] Error processing {file_path}: {str(e)}")
        import traceback
        logging.error(f"[Python_Multi] Traceback: {traceback.format_exc()}")
        return None

def process_tradingview_csv(file_path, date_col, date_fmt, pnl_cols, sheet_name=None):
    logging.info(f"[TradingView] Processing file: {file_path}")
    try:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            try:
                if sheet_name:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    logging.info(f"[TradingView] Reading sheet '{sheet_name}'")
                else:
                    df = pd.read_excel(file_path)
            except Exception as sheet_error:
                logging.warning(f"[TradingView] Could not read sheet '{sheet_name}': {sheet_error}")
                df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        if df.shape[1] >= 2:
            second_col = df.columns[1]
            df = df[df[second_col].astype(str).str.contains("Exit", case=False, na=False)]
        if df.empty:
            logging.warning(f"[TradingView] No 'Exit' rows found in {file_path}.")
            return None
        if date_col not in df.columns:
            logging.warning(f"[TradingView] Date column '{date_col}' not found in {file_path}.")
            return None
        parsed_dates = parse_multiple_date_formats(df[date_col], date_fmt)
        if parsed_dates is None:
            logging.warning(f"[TradingView] Failed to parse dates in {file_path}.")
            return None
        df[date_col] = parsed_dates.dt.strftime('%Y-%m-%d')
        df.dropna(subset=[date_col], inplace=True)
        pnl_cols_list = [col.strip() for col in pnl_cols.split(",")]
        pnl_col_found = None
        for col in pnl_cols_list:
            if col in df.columns:
                pnl_col_found = col
                break
        if pnl_col_found is None:
            logging.warning(f"[TradingView] No PNL column found among {pnl_cols_list} in {file_path}.")
            return None
        df.rename(columns={date_col: "Date", pnl_col_found: "PNL"}, inplace=True)
        df["Day"] = pd.to_datetime(df["Date"], errors="coerce").dt.day_name()
        strategy_name = os.path.splitext(os.path.basename(file_path))[0]
        df["Strategy"] = strategy_name
        df = df[["Strategy", "Date", "Day", "PNL"]]
        logging.info(f"[TradingView] Processed {len(df)} rows from {file_path}.")
        return df
    except Exception as e:
        logging.error(f"[TradingView] Error processing {file_path}: {e}")
        return None

def process_consolidated_xlsx(file_path, date_col, date_fmt, pnl_col, sheet_name):
    logging.info(f"[Consolidated] Processing file: {file_path}")
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name) if sheet_name else pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        if date_col not in df.columns:
            logging.warning(f"[Consolidated] Date column '{date_col}' not found in {file_path}.")
            return None
        date_fmt = date_fmt.replace("%%", "%")
        df[date_col] = pd.to_datetime(df[date_col], format=date_fmt, errors="coerce")
        df.dropna(subset=[date_col], inplace=True)
        if df.empty:
            logging.warning(f"[Consolidated] No valid dates in {file_path}.")
            return None
        df[date_col] = df[date_col].dt.strftime('%Y-%m-%d')
        if pnl_col not in df.columns:
            logging.warning(f"[Consolidated] PNL column '{pnl_col}' not found in {file_path}.")
            return None
        df.rename(columns={date_col: "Date", pnl_col: "PNL"}, inplace=True)
        df["Day"] = pd.to_datetime(df["Date"], errors="coerce").dt.day_name()
        strategy_name = os.path.splitext(os.path.basename(file_path))[0]
        df["Strategy"] = strategy_name
        df = df[["Strategy", "Date", "Day", "PNL"]]
        logging.info(f"[Consolidated] Processed {len(df)} rows from {file_path}.")
        return df
    except Exception as e:
        logging.error(f"[Consolidated] Error processing {file_path}: {e}")
        return None

def process_tradingview_zone_file(filepath, config_section):
    logging.info(f"[TV_Zone] Processing file: {filepath}")
    try:
        sheet_name = config_section.get("sheet_name", "List of trades")
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        df.columns = df.columns.str.strip()
        if len(df.columns) < 2:
            logging.warning(f"[TV_Zone] Not enough columns in {filepath}.")
            return None
        second_col = df.columns[1]
        df = df[df[second_col].astype(str).str.contains("entry", case=False, na=False)]
        if df.empty:
            logging.warning(f"[TV_Zone] No entry rows found in {filepath}.")
            return None
        date_col = config_section.get("date_column_name", "Date/Time")
        if date_col not in df.columns:
            logging.warning(f"[TV_Zone] Date column '{date_col}' not found in {filepath}.")
            return None
        df[date_col] = parse_multiple_date_formats(df[date_col], config_section.get("date_format", "%Y-%m-%d %H:%M:%S"))
        df.dropna(subset=[date_col], inplace=True)
        if df.empty:
            logging.warning(f"[TV_Zone] No valid dates after parsing in {filepath}.")
            return None

        def assign_zone(timestamp):
            zone_method = config_section.get("zone_method", "equal").strip().lower()
            market_open = datetime.datetime.strptime(config_section.get("market_open_time", "09:15:00"), "%H:%M:%S").time()
            market_close = datetime.datetime.strptime(config_section.get("market_close_time", "15:30:00"), "%H:%M:%S").time()
            open_dt = datetime.datetime.combine(timestamp.date(), market_open)
            close_dt = datetime.datetime.combine(timestamp.date(), market_close)
            if timestamp < open_dt or timestamp >= close_dt:
                return "Outside Market"
            if zone_method == "equal":
                num_zones = int(config_section.get("num_zones", "4"))
                total_seconds = (close_dt - open_dt).total_seconds()
                zone_duration = total_seconds / num_zones
                seconds_since_open = (timestamp - open_dt).total_seconds()
                zone_index = int(seconds_since_open // zone_duration) + 1
                return f"Zone {zone_index}"
            elif zone_method == "freeform":
                boundaries_str = config_section.get("freeform_zone_boundaries", "")
                if not boundaries_str:
                    return "Undefined"
                boundaries = [datetime.datetime.strptime(b.strip(), "%H:%M").time() for b in boundaries_str.split(",") if b.strip()]
                for i in range(len(boundaries) - 1):
                    start = datetime.datetime.combine(timestamp.date(), boundaries[i])
                    end = datetime.datetime.combine(timestamp.date(), boundaries[i+1])
                    if start <= timestamp < end:
                        return f"Zone {i+1}"
                return "Outside Market"
            elif zone_method == "custom":
                windows_str = config_section.get("custom_zone_windows", "")
                if not windows_str:
                    return "Undefined"
                windows = []
                for window in windows_str.split(","):
                    if "-" in window:
                        start_str, end_str = window.split("-")
                        try:
                            start = datetime.datetime.strptime(start_str.strip(), "%H:%M").time()
                            end = datetime.datetime.strptime(end_str.strip(), "%H:%M").time()
                            windows.append((start, end))
                        except Exception as e:
                            logging.error(f"Error parsing custom window '{window}': {e}")
                for i, (start, end) in enumerate(windows):
                    start_dt = datetime.datetime.combine(timestamp.date(), start)
                    end_dt = datetime.datetime.combine(timestamp.date(), end)
                    if start_dt <= timestamp < end_dt:
                        return f"Custom Zone {i+1}"
                return "Outside Market"
            else:
                return "Undefined"
        df["Zone"] = df[date_col].apply(lambda ts: assign_zone(ts))
        df["Date"] = df[date_col].dt.strftime("%Y-%m-%d")
        df["Day"] = df[date_col].dt.day_name()
        pnl_cols = config_section.get("pnl_column_name", "Profit INR,Profit USD,Profit,Profit_").split(",")
        pnl_col = pnl_cols[0].strip()
        if pnl_col not in df.columns:
            logging.warning(f"[TV_Zone] PNL column '{pnl_col}' not found in {filepath}.")
            return None
        df.rename(columns={pnl_col: "PNL"}, inplace=True)
        strategy_name = os.path.splitext(os.path.basename(filepath))[0]
        df["Strategy"] = strategy_name
        df = df[["Strategy", "Date", "Day", "Zone", "PNL"]]
        logging.info(f"[TV_Zone] Processed {len(df)} rows from {filepath}.")
        return df
    except Exception as e:
        logging.error(f"[TV_Zone] Error processing {filepath}: {e}")
        return None

# ------------------ New Processing Functions for FORMAT_8_PYTHON_MULTI_ZONE_XLSX ------------------

def process_python_multi_zone_xlsx(file_path, config_section):
    """
    Processes a single 'Python Multi-Zone' XLSX file.
    - Reads each sheet ending with "Trans"
    - Combines date and time columns (e.g., "Entry Date" and "Enter On") into a single datetime
    - Assigns a Zone based on market hours and zone method
    - Builds the Strategy name as follows:
         * If the workbook contains a sheet named "TV Setting" and a sheet named "Portfolio Trans":
               -> Process only the "Portfolio Trans" sheet and use the Excel file's base name as the strategy.
         * Otherwise, process sheets ending with "Trans" normally and build strategy names as <ExcelFileName>_<SheetNameWithoutTrans>.
    - Returns a DataFrame with columns: [Strategy, Date, Day, Zone, PNL]
    """
    logging.info(f"[Python_Multi_Zone] Processing file: {file_path}")
    try:
        excel_file = pd.ExcelFile(file_path)
        sheets = excel_file.sheet_names

        # Retrieve configuration values
        date_col = config_section.get("date_column_name", "Entry Date").strip()
        # Note: Updated to use "Enter On" instead of "Entry On"
        time_col = config_section.get("time_column_name", "Enter On").strip()
        pnl_col  = config_section.get("pnl_column_name", "Net PNL").strip()
        date_fmt = config_section.get("date_format", "%%Y-%%m-%%d").replace("%%", "%")
        time_fmt = config_section.get("time_format", "hh:mm:ss")  # Expected to be parsed as "%H:%M:%S"

        market_open_str  = config_section.get("market_open_time",  "09:15:00")
        market_close_str = config_section.get("market_close_time", "15:30:00")
        zone_method      = config_section.get("zone_method", "equal").strip().lower()
        num_zones        = int(config_section.get("num_zones", "4"))
        freeform_boundaries = config_section.get("freeform_zone_boundaries", "")
        custom_windows      = config_section.get("custom_zone_windows", "")

        # Get Excel file name (without extension) for strategy naming
        excel_basename = os.path.splitext(os.path.basename(file_path))[0]

        multi_data = []

        # Check if the workbook contains a sheet named "TV Setting"
        if any(sheet.strip().lower() == "tv setting" for sheet in sheets):
            # If yes, check if "Portfolio Trans" exists
            if any(sheet.strip().lower() == "portfolio trans" for sheet in sheets):
                logging.info(f"[Python_Multi_Zone] Detected 'TV Setting' and 'Portfolio Trans' in {file_path}.")
                sheet_to_process = next(sheet for sheet in sheets if sheet.strip().lower() == "portfolio trans")
                strategy_name = excel_basename  # Use the Excel file base name
                df = pd.read_excel(file_path, sheet_name=sheet_to_process)
                df.columns = df.columns.str.strip()
                if date_col not in df.columns or time_col not in df.columns:
                    logging.warning(f"[Python_Multi_Zone] Missing date/time columns '{date_col}'/'{time_col}' in sheet '{sheet_to_process}' of file '{file_path}'. Skipping.")
                elif pnl_col not in df.columns:
                    logging.warning(f"[Python_Multi_Zone] Missing PNL column '{pnl_col}' in sheet '{sheet_to_process}'. Skipping.")
                else:
                    df["EntryDatetime"] = pd.to_datetime(
                        df[date_col].astype(str) + " " + df[time_col].astype(str),
                        format=f"{date_fmt} {time_fmt}",
                        errors="coerce"
                    )
                    df.dropna(subset=["EntryDatetime"], inplace=True)
                    if not df.empty:
                        def assign_zone(ts):
                            if pd.isnull(ts):
                                return "Undefined"
                            mo = datetime.datetime.strptime(market_open_str, "%H:%M:%S").time()
                            mc = datetime.datetime.strptime(market_close_str, "%H:%M:%S").time()
                            open_dt = datetime.datetime.combine(ts.date(), mo)
                            close_dt = datetime.datetime.combine(ts.date(), mc)
                            if ts < open_dt or ts >= close_dt:
                                return "Outside Market"
                            if zone_method == "equal":
                                total_seconds = (close_dt - open_dt).total_seconds()
                                zone_duration = total_seconds / num_zones
                                seconds_offset = (ts - open_dt).total_seconds()
                                zone_index = int(seconds_offset // zone_duration) + 1
                                return f"Zone {zone_index}"
                            elif zone_method == "freeform":
                                if not freeform_boundaries.strip():
                                    return "Undefined"
                                boundary_strs = [b.strip() for b in freeform_boundaries.split(",") if b.strip()]
                                boundaries = [datetime.datetime.strptime(t, "%H:%M").time() for t in boundary_strs]
                                for i in range(len(boundaries)-1):
                                    start_dt = datetime.datetime.combine(ts.date(), boundaries[i])
                                    end_dt = datetime.datetime.combine(ts.date(), boundaries[i+1])
                                    if start_dt <= ts < end_dt:
                                        return f"Zone {i+1}"
                                return "Outside Market"
                            elif zone_method == "custom":
                                if not custom_windows.strip():
                                    return "Undefined"
                                windows = []
                                for window in custom_windows.split(","):
                                    window = window.strip()
                                    if "-" in window:
                                        start_str, end_str = window.split("-")
                                        try:
                                            start_t = datetime.datetime.strptime(start_str.strip(), "%H:%M").time()
                                            end_t = datetime.datetime.strptime(end_str.strip(), "%H:%M").time()
                                            windows.append((start_t, end_t))
                                        except Exception as e:
                                            logging.error(f"[Python_Multi_Zone] Bad custom window '{window}': {e}")
                                for i, (wstart, wend) in enumerate(windows):
                                    start_dt = datetime.datetime.combine(ts.date(), wstart)
                                    end_dt = datetime.datetime.combine(ts.date(), wend)
                                    if start_dt <= ts < end_dt:
                                        return f"Custom Zone {i+1}"
                                return "Outside Market"
                            else:
                                return "Undefined"
                        df["Zone"] = df["EntryDatetime"].apply(assign_zone)
                        df["Date"] = df["EntryDatetime"].dt.strftime("%Y-%m-%d")
                        df["Day"] = df["EntryDatetime"].dt.day_name()
                        df.rename(columns={pnl_col: "PNL"}, inplace=True)
                        df["Strategy"] = strategy_name
                        df = df[["Strategy", "Date", "Day", "Zone", "PNL"]]
                        multi_data.append(df)
            else:
                logging.info(f"[Python_Multi_Zone] 'TV Setting' found but no 'Portfolio Trans' in {file_path}. Reverting to normal processing.")

        # If no data collected via TV Setting condition, process sheets normally.
        if not multi_data:
            for sheet in sheets:
                if sheet.strip().lower() == "portfolio trans":
                    continue
                if sheet.strip().endswith("Trans"):
                    sheet_name_str = sheet.strip()
                    if sheet_name_str.lower().endswith("trans"):
                        sheet_name_str = sheet_name_str[:-5].strip()
                    strategy_name = f"{excel_basename}_{sheet_name_str}"
                    logging.info(f"[Python_Multi_Zone] Processing sheet '{sheet}' => Strategy '{strategy_name}'")
                    df = pd.read_excel(file_path, sheet_name=sheet)
                    df.columns = df.columns.str.strip()
                    if date_col not in df.columns or time_col not in df.columns:
                        logging.warning(f"[Python_Multi_Zone] Missing date/time columns '{date_col}'/'{time_col}' in sheet '{sheet}' of file '{file_path}'. Skipping.")
                        continue
                    if pnl_col not in df.columns:
                        logging.warning(f"[Python_Multi_Zone] Missing PNL column '{pnl_col}' in sheet '{sheet}'. Skipping.")
                        continue
                    df["EntryDatetime"] = pd.to_datetime(
                        df[date_col].astype(str) + " " + df[time_col].astype(str),
                        format=f"{date_fmt} {time_fmt}",
                        errors="coerce"
                    )
                    df.dropna(subset=["EntryDatetime"], inplace=True)
                    if df.empty:
                        logging.warning(f"[Python_Multi_Zone] No valid rows after parsing date/time in sheet '{sheet}'.")
                        continue
                    def assign_zone(ts):
                        if pd.isnull(ts):
                            return "Undefined"
                        mo = datetime.datetime.strptime(market_open_str, "%H:%M:%S").time()
                        mc = datetime.datetime.strptime(market_close_str, "%H:%M:%S").time()
                        open_dt = datetime.datetime.combine(ts.date(), mo)
                        close_dt = datetime.datetime.combine(ts.date(), mc)
                        if ts < open_dt or ts >= close_dt:
                            return "Outside Market"
                        if zone_method == "equal":
                            total_seconds = (close_dt - open_dt).total_seconds()
                            zone_duration = total_seconds / num_zones
                            seconds_offset = (ts - open_dt).total_seconds()
                            zone_index = int(seconds_offset // zone_duration) + 1
                            return f"Zone {zone_index}"
                        elif zone_method == "freeform":
                            if not freeform_boundaries.strip():
                                return "Undefined"
                            boundary_strs = [b.strip() for b in freeform_boundaries.split(",") if b.strip()]
                            boundaries = [datetime.datetime.strptime(t, "%H:%M").time() for t in boundary_strs]
                            for i in range(len(boundaries)-1):
                                start_dt = datetime.datetime.combine(ts.date(), boundaries[i])
                                end_dt = datetime.datetime.combine(ts.date(), boundaries[i+1])
                                if start_dt <= ts < end_dt:
                                    return f"Zone {i+1}"
                            return "Outside Market"
                        elif zone_method == "custom":
                            if not custom_windows.strip():
                                return "Undefined"
                            windows = []
                            for window in custom_windows.split(","):
                                window = window.strip()
                                if "-" in window:
                                    start_str, end_str = window.split("-")
                                    try:
                                        start_t = datetime.datetime.strptime(start_str.strip(), "%H:%M").time()
                                        end_t = datetime.datetime.strptime(end_str.strip(), "%H:%M").time()
                                        windows.append((start_t, end_t))
                                    except Exception as e:
                                        logging.error(f"[Python_Multi_Zone] Bad custom window '{window}': {e}")
                            for i, (wstart, wend) in enumerate(windows):
                                start_dt = datetime.datetime.combine(ts.date(), wstart)
                                end_dt = datetime.datetime.combine(ts.date(), wend)
                                if start_dt <= ts < end_dt:
                                    return f"Custom Zone {i+1}"
                            return "Outside Market"
                        else:
                            return "Undefined"
                    df["Zone"] = df["EntryDatetime"].apply(assign_zone)
                    df["Date"] = df["EntryDatetime"].dt.strftime("%Y-%m-%d")
                    df["Day"] = df["EntryDatetime"].dt.day_name()
                    df.rename(columns={pnl_col: "PNL"}, inplace=True)
                    df["Strategy"] = strategy_name
                    df = df[["Strategy", "Date", "Day", "Zone", "PNL"]]
                    multi_data.append(df)
        if not multi_data:
            logging.warning(f"[Python_Multi_Zone] No valid data found in file: {file_path}")
            return None
        combined_df = pd.concat(multi_data, ignore_index=True)
        logging.info(f"[Python_Multi_Zone] Processed {len(combined_df)} rows from file '{file_path}'")
        return combined_df
    except Exception as e:
        logging.error(f"[Python_Multi_Zone] Error processing file '{file_path}': {e}")
        return None

def run_python_multi_zone_only(config_path, input_root, output_root, bot_token=None, chat_id=None):
    """
    Processor function for FORMAT_8_PYTHON_MULTI_ZONE_XLSX.
    It reads the configuration, scans the input folder for Excel files,
    processes each file with process_python_multi_zone_xlsx, merges the data,
    pivots it based on merge keys (default: Date,Zone), and writes the output file.
    The final output file is named like: python_zone_consolidated_YYYYMMDD_HHMMSS.xlsx.
    """
    logging.info("Running Python Multi-Zone Processor...")
    config = configparser.ConfigParser(interpolation=None)
    if not os.path.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        return None
    config.read(config_path)

    section_key = "FORMAT_8_PYTHON_MULTI_ZONE_XLSX"
    if section_key not in config.sections():
        logging.warning(f"No [{section_key}] section found. Exiting.")
        return None
    cfg = config[section_key]
    folder = cfg.get("folder", "Python_Multi_Zone_Files").strip()

    input_dir = os.path.join(input_root, folder)
    file_paths = glob.glob(os.path.join(input_dir, "*.xls*"))
    if not file_paths:
        logging.warning(f"No Excel files found in {input_dir}.")
        return None

    update_progress(10, f"Processing {len(file_paths)} Python Multi-Zone Excel files")
    all_data = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_python_multi_zone_xlsx, fp, cfg) for fp in file_paths]
        for i, future in enumerate(as_completed(futures)):
            update_progress(10 + int(40 * i / len(futures)), f"Processed {i+1}/{len(futures)} files")
            df = future.result()
            if df is not None and not df.empty:
                all_data.append(df)

    if check_should_stop():
        logging.warning("Stop signal detected after processing. Exiting.")
        return None

    merge_keys = cfg.get("merge_keys", "").strip()
    if not merge_keys:
        merge_keys = "Date,Zone"

    output_file = build_summary_and_save(all_data, output_root, "python_zone_consolidated", merge_keys=merge_keys)

    if output_file and bot_token and chat_id:
        if isinstance(output_file, list):
            out_str = ", ".join([os.path.basename(f) for f in output_file])
        else:
            out_str = os.path.basename(output_file)
        msg = f"<b>Python Multi-Zone Consolidation Complete</b>\nProcessed {len(all_data)} files\nOutput: {out_str}"
        send_telegram_notification(bot_token, chat_id, msg)
    return output_file

# ------------------ Consolidation Functions ------------------

def to_long_format(df):
    if "Date" not in df.columns or "Day" not in df.columns:
        return pd.DataFrame()
    date_day_cols = ["Date", "Day"]
    strategy_cols = [col for col in df.columns if col not in date_day_cols]
    long_df = pd.melt(df, id_vars=date_day_cols, value_vars=strategy_cols,
                      var_name="Strategy", value_name="PNL")
    return long_df

def build_summary_and_save(all_data, output_root, output_prefix, merge_keys=""):
    if not all_data:
        logging.warning(f"No valid data found for {output_prefix}.")
        return None

    if check_should_stop():
        logging.warning("Stop signal detected during summary. Exiting.")
        return None

    long_dfs = []
    for df in all_data:
        if df is None or df.empty:
            continue
        if "Strategy" in df.columns and "PNL" in df.columns:
            long_dfs.append(df)
        else:
            melted = pd.melt(df, id_vars=["Date", "Day"], var_name="Strategy", value_name="PNL")
            long_dfs.append(melted)

    if not long_dfs:
        logging.warning(f"No valid dataframes to merge for {output_prefix}.")
        return None

    merged_df = pd.concat(long_dfs, ignore_index=True).sort_values("Date")
    if merged_df.empty:
        logging.warning("Merged DataFrame is empty. Exiting.")
        return None

    if "Zone" in merged_df.columns:
        if merge_keys:
            index_cols = [k.strip() for k in merge_keys.split(",")]
        else:
            index_cols = ["Date", "Zone"]
        valid_zone = merged_df["Zone"].apply(lambda z: z not in ["Undefined", "Outside Market"])
        if valid_zone.sum() / len(merged_df) < 0.1:
            logging.warning("Zone values are mostly undefined; falling back to merge keys ['Date','Day'].")
            index_cols = ["Date", "Day"]
    else:
        index_cols = ["Date", "Day"]

    update_progress(50, f"Creating pivot table for {len(merged_df)} rows")
    pivot_df = merged_df.pivot_table(index=index_cols,
                                     columns="Strategy", values="PNL",
                                     aggfunc="sum", fill_value=0, sort=False).reset_index()
    pivot_df.columns = [str(col) for col in pivot_df.columns]
    
    if output_prefix.lower().startswith("tv_zone") or output_prefix.lower().startswith("python_zone_consolidated"):
        if "Zone" in pivot_df.columns:
            pivot_df.insert(2, "Day", pd.to_datetime(pivot_df["Date"]).dt.day_name())
    else:
        if "Date" in pivot_df.columns and "Day" in pivot_df.columns:
            pivot_df["Day"] = pd.to_datetime(pivot_df["Date"]).dt.day_name()

    strategy_cols = [col for col in pivot_df.columns if col not in index_cols and col != "Day"]
    pivot_df = pivot_df[[col for col in pivot_df.columns if col not in strategy_cols] + strategy_cols]

    update_progress(80, f"Creating CSV file(s) with {len(strategy_cols)} strategies")
    optimize_dir = os.path.join(output_root, "Optimize")
    os.makedirs(optimize_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_prefix.lower().startswith("tv_zone") or output_prefix.lower().startswith("python_zone_consolidated"):
        MAX_STRATEGIES_PER_FILE = 500
        ROW_LIMIT = 50000
        output_files = []
        total_strats = len(strategy_cols)
        num_col_groups = (total_strats - 1) // MAX_STRATEGIES_PER_FILE + 1
        part_counter = 1
        for col_group in range(num_col_groups):
            start_col = col_group * MAX_STRATEGIES_PER_FILE
            end_col = start_col + MAX_STRATEGIES_PER_FILE
            current_strats = strategy_cols[start_col:end_col]
            fixed_cols = [c for c in pivot_df.columns if c in index_cols or c == "Day"]
            current_cols = fixed_cols + current_strats
            current_df = pivot_df[current_cols]
            n_rows = len(current_df)
            num_row_groups = (n_rows - 1) // ROW_LIMIT + 1
            for row_group in range(num_row_groups):
                start_row = row_group * ROW_LIMIT
                end_row = start_row + ROW_LIMIT
                current_chunk = current_df.iloc[start_row:end_row]
                part_filename = f"{output_prefix}_{timestamp}_part{part_counter}.csv"
                output_path = os.path.join(optimize_dir, part_filename)
                current_chunk.to_csv(output_path, index=False)
                logging.info(f"{output_prefix} consolidated part saved at {output_path}")
                output_files.append(output_path)
                part_counter += 1
        update_progress(100, f"Complete! Generated {len(output_files)} files.")
        return output_files
    else:
        output_file = os.path.join(optimize_dir, f"{output_prefix}_{timestamp}.csv")
        pivot_df.to_csv(output_file, index=False)
        logging.info(f"{output_prefix} consolidated file saved at {output_file}")
        update_progress(100, f"Complete! Output file: {os.path.basename(output_file)}")
        return output_file

# ------------------ Processor Runner Functions ------------------

def run_backinzo_only(config_path, input_root, output_root, bot_token=None, chat_id=None):
    logging.info("Running Backinzo CSV Processor...")
    config = configparser.ConfigParser(interpolation=None)
    if not os.path.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        return None
    config.read(config_path)
    section = "FORMAT_1_BACKINZO_CSV"
    if section not in config.sections():
        logging.warning(f"No [{section}] section found. Exiting.")
        return None
    folder = config[section].get("folder", "Backinzo_Files").strip()
    date_col = config[section].get("date_column_name", "Date").strip()
    pnl_col = config[section].get("pnl_column_name", "P/L").strip()
    date_fmt_pre = config[section].get("date_format1", "%d-%b-%Y")
    date_fmt_main = config[section].get("date_format2", "%d-%b-%Y")
    input_dir = os.path.join(input_root, folder)
    file_paths = glob.glob(os.path.join(input_dir, "*.csv"))
    if not file_paths:
        logging.warning(f"No CSV files found in {input_dir}.")
        return None
    update_progress(10, f"Pre-processing {len(file_paths)} Backinzo files")
    for i, fp in enumerate(file_paths):
        if check_should_stop():
            logging.warning("Stop signal detected during pre-processing. Exiting.")
            return None
        update_progress(10 + int(10 * i / len(file_paths)), f"Pre-processing file {i+1}/{len(file_paths)}")
        pre_reformat_backinzo_csv(fp, date_fmt_pre)
    update_progress(20, f"Processing {len(file_paths)} Backinzo files")
    all_data = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_backinzo_csv, fp, date_col, date_fmt_main, pnl_col) for fp in file_paths]
        for i, future in enumerate(as_completed(futures)):
            update_progress(20 + int(30 * i / len(futures)), f"Processed {i+1}/{len(futures)} files")
            df = future.result()
            if df is not None and not df.empty:
                all_data.append(df)
    if check_should_stop():
        logging.warning("Stop signal detected after processing. Exiting.")
        return None
    output_file = build_summary_and_save(all_data, output_root, "Backinzo_Consolidated")
    if output_file and bot_token and chat_id:
        msg = f"<b>Backinzo Consolidation Complete</b>\nProcessed {len(all_data)} files\nOutput: {os.path.basename(output_file) if isinstance(output_file, str) else ', '.join([os.path.basename(f) for f in output_file])}"
        send_telegram_notification(bot_token, chat_id, msg)
    return output_file

def run_backinzo_multi_files(config_path, input_root, output_root, bot_token=None, chat_id=None):
    logging.info("Running Backinzo Multi-File Processor...")
    config = configparser.ConfigParser(interpolation=None)
    if not os.path.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        return None
    config.read(config_path)
    section = "FORMAT_5_BACKINZO_Multi_CSV"
    if section not in config.sections():
        logging.warning(f"No [{section}] section found. Exiting.")
        return None
    folder = config[section].get("folder", "Backinzo_Multi_Files").strip()
    date_fmt_pre = config[section].get("date_format1", "%d-%b-%Y")
    date_fmt_main = config[section].get("date_format2", "%d-%b-%Y")
    input_dir = os.path.join(input_root, folder)
    file_paths = glob.glob(os.path.join(input_dir, "*.csv"))
    if not file_paths:
        logging.warning(f"No CSV files found in {input_dir}.")
        return None
    update_progress(10, f"Pre-processing {len(file_paths)} Backinzo multi files")
    for i, fp in enumerate(file_paths):
        if check_should_stop():
            logging.warning("Stop signal detected during pre-processing. Exiting.")
            return None
        update_progress(10 + int(10 * i / len(file_paths)), f"Pre-processing file {i+1}/{len(file_paths)}")
        pre_reformat_backinzo_multi_csv(fp, date_fmt_pre)
    update_progress(20, f"Processing {len(file_paths)} Backinzo multi files")
    all_data = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_backinzo_multi_file, fp, date_fmt_main) for fp in file_paths]
        for i, future in enumerate(as_completed(futures)):
            update_progress(20 + int(30 * i / len(futures)), f"Processed {i+1}/{len(futures)} files")
            df = future.result()
            if df is not None and not df.empty:
                all_data.append(df)
    if check_should_stop():
        logging.warning("Stop signal detected after processing. Exiting.")
        return None
    merged_df = all_data[0]
    for df in all_data[1:]:
        merged_df = pd.merge(merged_df, df, on=["Date", "Day"], how="outer")
    output_file = build_summary_and_save([merged_df], output_root, "Backinzo_multi_consolidated")
    if output_file and bot_token and chat_id:
        msg = f"<b>Backinzo Multi Consolidation Complete</b>\nProcessed {len(all_data)} files\nOutput: {os.path.basename(output_file) if isinstance(output_file, str) else ', '.join([os.path.basename(f) for f in output_file])}"
        send_telegram_notification(bot_token, chat_id, msg)
    return output_file

def run_python_only(config_path, input_root, output_root, bot_token=None, chat_id=None):
    logging.info("Running Python Excel Processor...")
    config = configparser.ConfigParser(interpolation=None)
    if not os.path.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        return None
    config.read(config_path)
    section = "FORMAT_2_PYTHON_XLSX"
    if section not in config.sections():
        logging.warning(f"No [{section}] section found. Exiting.")
        return None
    folder = config[section].get("folder", "Python_Files").strip()
    sheet_name = config[section].get("sheet_name", "").strip()
    date_col = config[section].get("date_column_name", "Exit Date").strip()
    date_fmt = config[section].get("date_format", "%%Y-%%m-%%d")
    pnl_col = config[section].get("pnl_column_name", "Net PNL").strip()
    input_dir = os.path.join(input_root, folder)
    file_paths = glob.glob(os.path.join(input_dir, "*.xls*"))
    if not file_paths:
        logging.warning(f"No Excel files found in {input_dir}.")
        return None
    update_progress(10, f"Processing {len(file_paths)} Python Excel files")
    all_data = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_python_xlsx, fp, date_col, date_fmt, pnl_col, sheet_name) for fp in file_paths]
        for i, future in enumerate(as_completed(futures)):
            update_progress(10 + int(40 * i / len(futures)), f"Processed {i+1}/{len(futures)} files")
            df = future.result()
            if df is not None and not df.empty:
                all_data.append(df)
    if check_should_stop():
        logging.warning("Stop signal detected after processing. Exiting.")
        return None
    output_file = build_summary_and_save(all_data, output_root, "Python_Consolidated")
    if output_file and bot_token and chat_id:
        msg = f"<b>Python Excel Consolidation Complete</b>\nProcessed {len(all_data)} files\nOutput: {os.path.basename(output_file) if isinstance(output_file, str) else ', '.join([os.path.basename(f) for f in output_file])}"
        send_telegram_notification(bot_token, chat_id, msg)
    return output_file

def run_python_multi_only(config_path, input_root, output_root, bot_token=None, chat_id=None):
    logging.info("Running Python Multi-File Processor...")
    config = configparser.ConfigParser(interpolation=None)
    if not os.path.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        return None
    config.read(config_path)
    section = "FORMAT_6_PYTHON_MULTI_XLSX"
    if section not in config.sections():
        logging.warning(f"No [{section}] section found. Exiting.")
        return None
    folder = config[section].get("folder", "Python_Multi_Files").strip()
    date_col = config[section].get("date_column_name", "Exit Date").strip()
    date_fmt = config[section].get("date_format", "%%Y-%%m-%%d")
    pnl_col = config[section].get("pnl_column_name", "Net PNL").strip()
    input_dir = os.path.join(input_root, folder)
    file_paths = glob.glob(os.path.join(input_dir, "*.xls*"))
    if not file_paths:
        logging.warning(f"No Excel files found in {input_dir}.")
        return None
    update_progress(10, f"Processing {len(file_paths)} Python multi Excel files")
    all_data = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_python_multi_xlsx, fp, date_col, date_fmt, pnl_col) for fp in file_paths]
        for i, future in enumerate(as_completed(futures)):
            update_progress(10 + int(40 * i / len(futures)), f"Processed {i+1}/{len(futures)} files")
            df = future.result()
            if df is not None and not df.empty:
                all_data.append(df)
    if check_should_stop():
        logging.warning("Stop signal detected after processing. Exiting.")
        return None
    output_file = build_summary_and_save(all_data, output_root, "Python_Multi_Consolidated")
    if output_file and bot_token and chat_id:
        msg = f"<b>Python Multi Excel Consolidation Complete</b>\nProcessed {len(all_data)} files\nOutput: {os.path.basename(output_file) if isinstance(output_file, str) else ', '.join([os.path.basename(f) for f in output_file])}"
        send_telegram_notification(bot_token, chat_id, msg)
    return output_file

def run_tradingview_only(config_path, input_root, output_root, bot_token=None, chat_id=None):
    logging.info("Running TradingView Processor...")
    config = configparser.ConfigParser(interpolation=None)
    if not os.path.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        return None
    config.read(config_path)
    section = "FORMAT_3_TRADING_VIEW_CSV"
    if section not in config.sections():
        logging.warning(f"No [{section}] section found. Exiting.")
        return None
    folder = config[section].get("folder", "TV_Files").strip()
    date_col = config[section].get("date_column_name", "Date/Time").strip()
    date_fmt = config[section].get("date_format", "%%Y-%%m-%%d %%H:%%M:%%S,%%d-%%m-%%Y %%H:%%M:%%S")
    pnl_cols = config[section].get("pnl_column_name", "Profit INR,Profit USD,Profit_,Profit").strip()
    sheet_name = config[section].get("sheet_name", "").strip()
    input_dir = os.path.join(input_root, folder)
    file_paths = glob.glob(os.path.join(input_dir, "*.csv"))
    file_paths.extend(glob.glob(os.path.join(input_dir, "*.xls*")))
    if not file_paths:
        logging.warning(f"No TradingView files found in {input_dir}.")
        return None
    update_progress(10, f"Processing {len(file_paths)} TradingView files")
    all_data = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_tradingview_csv, fp, date_col, date_fmt, pnl_cols, sheet_name) for fp in file_paths]
        for i, future in enumerate(as_completed(futures)):
            update_progress(10 + int(40 * i / len(futures)), f"Processed {i+1}/{len(futures)} files")
            df = future.result()
            if df is not None and not df.empty:
                all_data.append(df)
    if check_should_stop():
        logging.warning("Stop signal detected after processing. Exiting.")
        return None
    output_file = build_summary_and_save(all_data, output_root, "TradingView_Consolidated")
    if output_file and bot_token and chat_id:
        msg = f"<b>TradingView Consolidation Complete</b>\nProcessed {len(all_data)} files\nOutput: {os.path.basename(output_file) if isinstance(output_file, str) else ', '.join([os.path.basename(f) for f in output_file])}"
        send_telegram_notification(bot_token, chat_id, msg)
    return output_file

def run_consolidated_only(config_path, input_root, output_root, bot_token=None, chat_id=None):
    logging.info("Running Consolidated Excel Processor...")
    config = configparser.ConfigParser(interpolation=None)
    if not os.path.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        return None
    config.read(config_path)
    section = "FORMAT_4_CONSOLIDATED_XLSX"
    if section not in config.sections():
        logging.warning(f"No [{section}] section found. Exiting.")
        return None
    folder = config[section].get("folder", "Consolidated_Files").strip()
    sheet_name = config[section].get("sheet_name", "").strip()
    date_col = config[section].get("date_column_name", "Date").strip()
    date_fmt = config[section].get("date_format", "%%d-%%m-%%Y")
    pnl_col = config[section].get("pnl_column_name", "Profit").strip()
    input_dir = os.path.join(input_root, folder)
    file_paths = glob.glob(os.path.join(input_dir, "*.xls*"))
    if not file_paths:
        logging.warning(f"No Consolidated Excel files found in {input_dir}.")
        return None
    update_progress(10, f"Processing {len(file_paths)} Consolidated Excel files")
    all_data = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_consolidated_xlsx, fp, date_col, date_fmt, pnl_col, sheet_name) for fp in file_paths]
        for i, future in enumerate(as_completed(futures)):
            update_progress(10 + int(40 * i / len(futures)), f"Processed {i+1}/{len(futures)} files")
            df = future.result()
            if df is not None and not df.empty:
                all_data.append(df)
    if check_should_stop():
        logging.warning("Stop signal detected after processing. Exiting.")
        return None
    output_file = build_summary_and_save(all_data, output_root, "Consolidated_Excel")
    if output_file and bot_token and chat_id:
        msg = f"<b>Consolidated Excel Consolidation Complete</b>\nProcessed {len(all_data)} files\nOutput: {os.path.basename(output_file) if isinstance(output_file, str) else ', '.join([os.path.basename(f) for f in output_file])}"
        send_telegram_notification(bot_token, chat_id, msg)
    return output_file

def run_tradingview_zone_only(config_path, input_root, output_root, bot_token=None, chat_id=None):
    logging.info("Running TradingView Zone Processor...")
    config = configparser.ConfigParser(interpolation=None)
    if not os.path.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        return None
    config.read(config_path)
    section = "FORMAT_7_TradingView_Zone"
    if section not in config.sections():
        logging.warning(f"No [{section}] section found. Exiting.")
        return None
    cfg = config[section]
    folder = cfg.get("folder", "TV_Zone_Files").strip()
    input_dir = os.path.join(input_root, folder)
    file_paths = glob.glob(os.path.join(input_dir, "*.xlsx"))
    if not file_paths:
        logging.warning(f"No TradingView Zone files found in {input_dir}.")
        return None
    update_progress(10, f"Processing {len(file_paths)} TradingView Zone files")
    all_data = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_tradingview_zone_file, fp, cfg) for fp in file_paths]
        for i, future in enumerate(as_completed(futures)):
            update_progress(10 + int(40 * i / len(futures)), f"Processed {i+1}/{len(futures)} files")
            df = future.result()
            if df is not None and not df.empty:
                all_data.append(df)
    if check_should_stop():
        logging.warning("Stop signal detected after processing. Exiting.")
        return None
    merge_keys = cfg.get("merge_keys", "").strip()
    if not merge_keys:
        merge_keys = "Date,Zone"
    output_file = build_summary_and_save(all_data, output_root, "TV_Zone_Consolidated", merge_keys=merge_keys)
    if output_file and bot_token and chat_id:
        if isinstance(output_file, list):
            out_str = ", ".join([os.path.basename(f) for f in output_file])
        else:
            out_str = os.path.basename(output_file)
        msg = f"<b>TradingView Zone Consolidation Complete</b>\nProcessed {len(all_data)} files\nOutput: {out_str}"
        send_telegram_notification(bot_token, chat_id, msg)
    return output_file

def run_all_processors(config_path, input_root, output_root, bot_token=None, chat_id=None):
    logging.info("Running all processors...")
    update_progress(0, "Starting all processors")
    output_files = []
    processors = [
        ("Backinzo", run_backinzo_only),
        ("Python", run_python_only),
        ("TradingView", run_tradingview_only),
        ("Consolidated", run_consolidated_only),
        ("Backinzo_Multi", run_backinzo_multi_files),
        ("Python_Multi", run_python_multi_only),
        ("TV_Zone", run_tradingview_zone_only),
        ("Python_Multi_Zone", run_python_multi_zone_only)  # New format added here
    ]
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(func, config_path, input_root, output_root, bot_token, chat_id): name for name, func in processors}
        for i, future in enumerate(as_completed(futures)):
            proc_name = futures[future]
            update_progress(20 + int(70 * i / len(futures)), f"Completed processor: {proc_name}")
            result = future.result()
            if result:
                output_files.append(result)
    update_progress(100, f"All processors completed. Generated {len(output_files)} files.")
    if output_files and bot_token and chat_id:
        def flatten(files):
            if isinstance(files, list):
                return [f for item in files for f in (flatten(item) if isinstance(item, list) else [item])]
            else:
                return [files]
        flat_outputs = flatten(output_files)
        outputs_str = "\n".join([f"• {os.path.basename(f)}" for f in flat_outputs])
        msg = f"<b>All Processors Complete</b>\nGenerated {len(flat_outputs)} output files:\n{outputs_str}"
        send_telegram_notification(bot_token, chat_id, msg)
    return output_files

# ------------------ Main Entry Point ------------------

if __name__ == "__main__":
    bot_token = None
    chat_id = None
    if len(sys.argv) >= 3:
        bot_token = sys.argv[1]
        chat_id = sys.argv[2]
    outputs = run_all_processors(CONFIG_PATH, INPUT_DIR, OUTPUT_DIR, bot_token, chat_id)
    if outputs:
        logging.info("Consolidation complete. Output files:")
        for f in outputs:
            if isinstance(f, list):
                for part in f:
                    logging.info(f" - {part}")
            else:
                logging.info(f" - {f}")
    else:
        logging.info("No output files were generated.")

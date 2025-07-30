#!/usr/bin/env python

"""
Multi-Strategy Portfolio Optimizer (Local Version with Zone-Based Mode)
-------------------------------------------------------------------------
This script implements a robust portfolio optimizer that supports both the
traditional entire‑day mode and a new zone‑based mode. In addition to all
existing functionality, a new configuration parameter "use_checkpoint"
controls whether optimization runs resume from saved checkpoint files or
start fresh.
"""

import os
import sys
import time
import logging
import traceback
import random
import pickle
from datetime import datetime
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional, Callable, Union

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# --------------------------
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable for TensorFlow logging
tf_log_count = 0

def log_tf_missing(message="TensorFlow not installed. GPU acceleration will be disabled."):
    global tf_log_count
    if tf_log_count < 1:
        logging.info(message)
        tf_log_count += 1

# Optional GPU setup with TensorFlow
try:
    import tensorflow as tf
    tf_imported = True
except ImportError:
    tf = None
    tf_imported = False
    log_tf_missing()

import requests
from concurrent.futures import ProcessPoolExecutor, as_completed

# =============================================================================
# Global Variables and Constants
# =============================================================================
BALANCED_MODE: bool = False
FILE_LABELS: List[str] = []
DESIRED_RATIO: Dict[str, int] = {}
PENALTY_FACTOR: float = 1.0
USE_CHECKPOINT: bool = True  # This will be set from the config file later
CHECKPOINT_PER_SIZE: bool = True  # This will be set from the config file later
PRESERVE_CHECKPOINTS: bool = True  # This will be set from the config file later
# Fitness calibration parameters
MAX_CONSISTENCY: float = 5.0  # Default values, will be updated by calibration
MEAN_CONSISTENCY: float = 1.0
MAX_CORRELATION: float = 0.8
MEAN_CORRELATION: float = 0.4

# Global data storage for multiprocessing
GLOBAL_DAILY_MATRIX = None
GLOBAL_CORR_MATRIX = None
GLOBAL_METRIC = None
GLOBAL_DRAWDOWN_THRESHOLD = 0.0

# =============================================================================
# Environment Setup
# =============================================================================
def setup_environment() -> bool:
    if tf is None:
        logging.info("TensorFlow not installed, skipping GPU configuration. Running on CPU.")
        cpu_count = os.cpu_count() or 1
        logging.info(f"CPU Cores available: {cpu_count}")
        os.environ["OMP_NUM_THREADS"] = str(cpu_count)
        os.environ["MKL_NUM_THREADS"] = str(cpu_count)
        pd.set_option('compute.use_bottleneck', True)
        pd.set_option('compute.use_numexpr', True)
        return False
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"✅ GPU acceleration enabled: {len(gpus)} GPU(s) available")
            logging.info(f"   GPU Device: {tf.test.gpu_device_name()}")
            return True
        else:
            logging.info("⚠️ No GPU found. Using CPU for computations.")
        cpu_count = os.cpu_count() or 1
        logging.info(f"CPU Cores available: {cpu_count}")
        os.environ["OMP_NUM_THREADS"] = str(cpu_count)
        os.environ["MKL_NUM_THREADS"] = str(cpu_count)
        pd.set_option('compute.use_bottleneck', True)
        pd.set_option('compute.use_numexpr', True)
        return False
    except Exception as e:
        logging.error(f"Error setting up environment: {e}")
        return False

# =============================================================================
# Telegram Alert Helper (unchanged)
# =============================================================================
def send_telegram_alert(message: str, bot_token: str = '', chat_id: str = '',
                        max_retries: int = 3, retry_delay: int = 5) -> bool:
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, data=data, timeout=10)
            resp.raise_for_status()
            logging.info("Telegram alert sent successfully.")
            return True
        except Exception as e:
            logging.warning(f"Telegram alert attempt {attempt+1} failed: {e}")
            time.sleep(retry_delay)
    logging.error(f"Failed to send Telegram alert after {max_retries} attempts")
    return False

# -----------------------------------------------------------------------------
# Safe Config Value Retrieval
# -----------------------------------------------------------------------------
def safe_get_config_value(config, section, key, default_value, conv_func=str):
    try:
        if section in config and key in config[section]:
            return conv_func(config[section][key])
        return default_value
    except (ValueError, TypeError) as e:
        logging.warning(f"Error converting {section}.{key}: {e}. Using default: {default_value}")
        return default_value

# =============================================================================
# Checkpoint Helper Functions
# =============================================================================
def get_checkpoint_filename(algorithm: str, filepath: str, portfolio_size: int = None) -> str:
    """
    Generate a checkpoint filename based on algorithm and optionally portfolio size.
    """
    if filepath is None:
        filepath = "."
    base_name = f"checkpoint_{algorithm.lower()}"
    if CHECKPOINT_PER_SIZE and portfolio_size is not None:
        base_name = f"{base_name}_size{portfolio_size}"
    if os.path.isdir(filepath):
        return os.path.join(filepath, f"{base_name}.pkl")
    directory = os.path.dirname(filepath)
    return os.path.join(directory, f"{base_name}.pkl")

def save_checkpoint(state: Dict[str, Any], algorithm: str, filepath: str, portfolio_size: int = None):
    if not USE_CHECKPOINT:
        return
    try:
        checkpoint_path = get_checkpoint_filename(algorithm, filepath, portfolio_size)
        with open(checkpoint_path, "wb") as f:
            pickle.dump(state, f)
        logging.info(f"Checkpoint saved: {checkpoint_path}")
    except Exception as e:
        logging.error(f"Error saving checkpoint: {e}")

def load_checkpoint(algorithm: str, filepath: str, portfolio_size: int = None) -> Dict[str, Any]:
    if not USE_CHECKPOINT:
        return {}
    try:
        checkpoint_path = get_checkpoint_filename(algorithm, filepath, portfolio_size)
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                state = pickle.load(f)
            logging.info(f"Checkpoint loaded: {checkpoint_path}")
            return state
        else:
            logging.info(f"No checkpoint found at {checkpoint_path}")
            return {}
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        return {}

def clean_checkpoints(directory: str):
    if PRESERVE_CHECKPOINTS:
        return
    try:
        for filename in os.listdir(directory):
            if filename.startswith("checkpoint_") and filename.endswith(".pkl"):
                filepath = os.path.join(directory, filename)
                os.remove(filepath)
                logging.info(f"Removed checkpoint file: {filepath}")
    except Exception as e:
        logging.error(f"Error cleaning checkpoint files: {e}")

# =============================================================================
# Data Loading Functions
# =============================================================================
def load_maxoi_data(filepath: str, strat_limit: Optional[Union[str, int]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info(f"Loading data from: {filepath}")
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Try to read as CSV first, then Excel if that fails
        try:
            df = pd.read_csv(filepath, header=0)
            logging.info(f"Successfully loaded CSV file: {filepath}")
        except Exception as csv_error:
            logging.info(f"Failed to load as CSV, trying Excel: {csv_error}")
            df = pd.read_excel(filepath, header=0)
            logging.info(f"Successfully loaded Excel file: {filepath}")
        
        summary_df = pd.DataFrame()  # No separate summary in this format
        daily_df = df.copy()
        if strat_limit is not None and strat_limit != 'all':
            try:
                strat_limit = int(strat_limit)
                if strat_limit > 0:
                    daily_df = daily_df.iloc[:, :3+strat_limit]
                else:
                    logging.warning(f"Invalid strat_limit {strat_limit}, using 'all'")
            except (ValueError, TypeError):
                logging.warning(f"Invalid strat_limit {strat_limit}, using 'all'")
        missing_count = daily_df.isna().sum().sum()
        if missing_count > 0:
            logging.warning(f"File '{filepath}' contains {missing_count} missing values. Filling with zeros.")
            daily_df = daily_df.fillna(0)
        return summary_df, daily_df
    except Exception as e:
        logging.error(f"Error loading file '{filepath}': {e}")
        raise

def load_and_merge_files(folder: str, strat_limit: Optional[Union[str, int]] = None) -> Tuple[pd.DataFrame, List[str]]:
    try:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Directory '{folder}' not found")
        
        # Look for CSV files first, then Excel files
        csv_files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
        xlsx_files = [f for f in os.listdir(folder) if f.lower().endswith(".xlsx")]
        files = csv_files + xlsx_files
        
        if not files:
            raise FileNotFoundError(f"No .csv or .xlsx files found in '{folder}' folder.")
        
        logging.info(f"Found {len(files)} files ({len(csv_files)} CSV, {len(xlsx_files)} Excel): {', '.join(files)}")
        dfs: List[pd.DataFrame] = []
        file_labels: List[str] = []
        for filename in tqdm(files, desc="Loading files"):
            filepath = os.path.join(folder, filename)
            try:
                _, daily_df = load_maxoi_data(filepath, strat_limit)
                num_strats = len(daily_df.columns) - 3
                logging.info(f"Selected file: {filename} with {num_strats} strategy columns")
                prefix = os.path.splitext(filename)[0]
                new_cols = list(daily_df.columns[:3]) + [f"{prefix}_{col}" for col in daily_df.columns[3:]]
                daily_df.columns = new_cols
                dfs.append(daily_df)
                file_labels.extend([prefix] * (len(daily_df.columns) - 3))
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
        if not dfs:
            raise ValueError("No valid data frames loaded.")
        merged_df = dfs[0]
        for i, df in enumerate(dfs[1:], 1):
            logging.info(f"Merging dataframe {i+1}/{len(dfs)}")
            try:
                merged_df = pd.merge(merged_df, df, on=["Date", "Zone", "Day"], how="outer", suffixes=("", "_dup"))
                dup_cols = [col for col in merged_df.columns if col.endswith("_dup")]
                for col in dup_cols:
                    base_col = col.replace("_dup", "")
                    if base_col in merged_df.columns:
                        merged_df[base_col] = merged_df[base_col].fillna(merged_df[col])
                    merged_df.drop(columns=[col], inplace=True)
            except Exception as e:
                logging.error(f"Error merging dataframe {i+1}: {e}")
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        if merged_df.isna().sum().sum() > 0:
            logging.warning("Merged dataframe contains missing values. Filling with zeros.")
            merged_df = merged_df.fillna(0)
        logging.info(f"Merged dataframe shape: {merged_df.shape}")
        return merged_df, file_labels
    except Exception as e:
        logging.error(f"Error in load_and_merge_files: {e}")
        raise

def load_consolidated_df_from_directory(consolidated_dir: str) -> pd.DataFrame:
    # Look for CSV files first, then Excel files
    csv_files = [os.path.join(consolidated_dir, f) for f in os.listdir(consolidated_dir)
                 if f.lower().endswith(".csv")]
    xlsx_files = [os.path.join(consolidated_dir, f) for f in os.listdir(consolidated_dir)
                  if f.lower().endswith(".xlsx")]
    files = csv_files + xlsx_files
    
    if not files:
        raise ValueError("No .csv or .xlsx files found in the directory.")
    
    logging.info(f"Selected consolidated files: {files}")
    df_list = []
    for file in files:
        try:
            # Try to read as CSV first
            if file.lower().endswith('.csv'):
                try:
                    df = pd.read_csv(file, header=0)
                    df_list.append(df)
                    logging.info(f"Successfully loaded CSV file: {file}")
                except Exception as e:
                    logging.error(f"Failed to read CSV file {file}: {e}")
            else:
                # Try with different Excel engines
                for engine in ['openpyxl', 'xlrd', 'odf']:
                    try:
                        df = pd.read_excel(file, header=0, engine=engine)
                        df_list.append(df)
                        logging.info(f"Successfully loaded Excel file {file} with {engine} engine")
                        break
                    except Exception as e:
                        logging.warning(f"Failed to read {file} with {engine} engine: {e}")
                        if engine == 'odf':  # Last engine to try
                            logging.error(f"Could not read {file} with any Excel engine")
        except Exception as e:
            logging.error(f"Failed to process {file}: {e}")
    if not df_list:
        raise ValueError("Could not read any files from the directory.")
    return pd.concat(df_list, ignore_index=True)

def build_zone_matrix_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    df["Zone"] = df["Zone"].astype(str)
    df = df[~df["Zone"].isin({"Outside Market", "nan", ""})]
    df.fillna(0, inplace=True)
    group_cols = ["Date", "Zone", "DAY"] if "DAY" in df.columns else ["Date", "Zone", "Day"]
    df.sort_values(group_cols, inplace=True)
    strategy_cols = [col for col in df.columns if col not in group_cols]
    all_zones = sorted(df["Zone"].unique())
    logging.info(f"All zones used for optimization: {all_zones}")
    grouped = df.groupby("Date")
    zone_matrix_list = []
    for date, group in grouped:
        group["Zone"] = group["Zone"].astype(str)
        zone_data_array = np.zeros((len(all_zones), len(strategy_cols)), dtype=float)
        for idx, zone in enumerate(all_zones):
            zone_group = group[group["Zone"] == zone]
            if not zone_group.empty:
                zone_values = zone_group[strategy_cols].sum().values.astype(float)
                zone_data_array[idx, :] = zone_values
        zone_matrix_list.append(zone_data_array)
    zone_matrix = np.stack(zone_matrix_list, axis=0)
    return zone_matrix, strategy_cols

def calibrate_zone_fitness_parameters(zone_matrix: np.ndarray) -> Dict[str, float]:
    """
    Dummy calibration of zone fitness parameters.
    Returns default values for max_consistency, mean_consistency, max_correlation, and mean_correlation.
    """
    return {
        "max_consistency": 5.0,
        "mean_consistency": 1.0,
        "max_correlation": 0.8,
        "mean_correlation": 0.4
    }



# =============================================================================
# ULTA Logic Functions 
# =============================================================================
def invert_trades(strategy_returns: np.ndarray) -> np.ndarray:
    return -strategy_returns

def calculate_ratio(roi: float, drawdown: float) -> float:
    loss = abs(drawdown)
    return roi / loss if loss != 0 else float('inf')

def apply_ulta_logic(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    inverted_strategies: Dict[str, Dict[str, float]] = {}
    updated_data = data.copy()
    new_columns = {}
    columns_to_drop = []
    for col in tqdm(data.columns[3:], desc="Applying ULTA logic"):
        try:
            arr = pd.to_numeric(data[col], errors='coerce').fillna(0).values
            roi = np.sum(arr)
            dd = np.min(arr)
            ratio = calculate_ratio(roi, dd)
            if ratio < 0:
                inv = invert_trades(arr)
                inv_roi = np.sum(inv)
                inv_dd = np.min(inv)
                inv_ratio = calculate_ratio(inv_roi, inv_dd)
                if inv_ratio > ratio:
                    inv_col = f"{col}_inv"
                    new_columns[inv_col] = inv
                    columns_to_drop.append(col)
                    inverted_strategies[col] = {
                        "original_roi": roi,
                        "inverted_roi": inv_roi,
                        "original_drawdown": dd,
                        "inverted_drawdown": inv_dd,
                        "original_ratio": ratio,
                        "inverted_ratio": inv_ratio
                    }
                    logging.debug(f"Inverted strategy {col}: ratio improved from {ratio:.2f} to {inv_ratio:.2f}")
        except Exception as e:
            logging.error(f"Error applying ULTA logic to column {col}: {e}")
            continue
    if columns_to_drop:
        updated_data.drop(columns=columns_to_drop, inplace=True)
    if new_columns:
        new_cols_df = pd.DataFrame(new_columns, index=updated_data.index)
        updated_data = pd.concat([updated_data, new_cols_df], axis=1)
    return updated_data, inverted_strategies

def generate_inversion_report(inverted_strategies: Dict[str, Dict[str, float]]) -> str:
    report_lines = [
        "# Inversion Report", "",
        "The following strategies had their returns inverted based on ULTA logic:"
    ]
    for strat, details in inverted_strategies.items():
        report_lines.append(f"## {strat}")
        report_lines.append(f"- **Original ROI:** {details['original_roi']:.2f}")
        report_lines.append(f"- **Inverted ROI:** {details['inverted_roi']:.2f}")
        report_lines.append(f"- **Original Drawdown:** {details['original_drawdown']:.2f}")
        report_lines.append(f"- **Inverted Drawdown:** {details['inverted_drawdown']:.2f}")
        report_lines.append(f"- **Original Ratio:** {details['original_ratio']:.2f}")
        report_lines.append(f"- **Inverted Ratio:** {details['inverted_ratio']:.2f}")
        report_lines.append("")
    return "\n".join(report_lines)

def plot_inversion_comparison(data: pd.DataFrame, strat_name: str, inversion_info: Dict[str, Dict[str, float]]) -> Optional[plt.Figure]:
    inv_col = f"{strat_name}_inv"
    if inv_col not in data.columns:
        return None
    try:
        original_returns = pd.to_numeric(data[strat_name], errors='coerce').fillna(0)
    except Exception:
        original_returns = np.zeros(len(data))
    try:
        inverted_returns = pd.to_numeric(data[inv_col], errors='coerce').fillna(0)
    except Exception:
        inverted_returns = np.zeros(len(data))
    original_curve = np.cumsum(original_returns)
    inverted_curve = np.cumsum(inverted_returns)
    try:
        fig = plt.figure(figsize=(10, 5))
    except Exception as e:
        logging.error(f"Error in plotting or processing: {e}")
        fig = None
    plt.plot(original_curve, label=f"Original {strat_name}")
    plt.plot(inverted_curve, label=f"Inverted {strat_name}")
    plt.title(f"Inversion Comparison for {strat_name}")
    plt.xlabel("Days")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    return fig

# =============================================================================
def write_inversion_report(report: str, output_dir: str):
    """Write the inversion report to a markdown file."""
    report_file = os.path.join(output_dir, "inversion_report.md")
    try:
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        logging.info("Inversion report written to {}".format(report_file))
    except Exception as e:
        logging.error("Error writing inversion report: {}".format(e))

def generate_and_save_performance_report(returns: np.ndarray, portfolio_name: str, output_dir: str):
    """Generate performance report metrics and save to a text file (images are already generated)."""
    report = generate_performance_report(returns, portfolio_name, output_dir)
    report_file = os.path.join(output_dir, "performance_report_{}.txt".format(portfolio_name))
    try:
        with open(report_file, "w", encoding="utf-8") as f:
            for key, value in report.get("metrics", {}).items():
                f.write("{}: {}\n".format(key, value))
        logging.info("Performance report saved to {}".format(report_file))
    except Exception as e:
        logging.error("Error writing performance report: {}".format(e))

# Enhanced Reporting Functions for Performance Metrics
# =============================================================================
def generate_performance_report(returns: np.ndarray, portfolio_name: str, output_dir: str) -> Dict[str, Any]:
    try:
        equity_curve = np.cumsum(returns)
        total_roi = np.sum(returns)
        peak = np.maximum.accumulate(equity_curve)
        max_drawdown = np.max(peak - equity_curve) if len(equity_curve) > 0 else 0
        win_days = np.sum(returns > 0)
        total_days = len(returns)
        win_percentage = win_days / total_days if total_days > 0 else 0
        pos_sum = np.sum(returns[returns > 0]) if any(returns > 0) else 0
        neg_sum = abs(np.sum(returns[returns < 0])) if any(returns < 0) else 0
        profit_factor = pos_sum / neg_sum if neg_sum != 0 else np.inf

        avg_win = np.mean(returns[returns > 0]) if any(returns > 0) else 0
        avg_loss = np.mean(abs(returns[returns < 0])) if any(returns < 0) else 0
        expectancy = (win_percentage * avg_win) - ((1 - win_percentage) * avg_loss)

        annual_return = total_roi * (252 / total_days) if total_days > 0 else 0
        daily_std = np.std(returns)
        annualized_std = daily_std * np.sqrt(252) if daily_std > 0 else 0
        sharpe_ratio = annual_return / annualized_std if annualized_std > 0 else 0
        sortino_ratio = annual_return / (np.std(returns[returns < 0]) * np.sqrt(252)) if any(returns < 0) else 0

        fig = plt.figure(figsize=(12, 6))
        plt.plot(equity_curve)
        plt.title(f"Equity Curve - {portfolio_name}")
        plt.xlabel("Days")
        plt.ylabel("Cumulative Return")
        plt.grid(True)

        plt.figure(figsize=(12, 6))
        plt.plot(peak - equity_curve)
        plt.title(f"Drawdowns - {portfolio_name}")
        plt.xlabel("Days")
        plt.ylabel("Drawdown")
        plt.grid(True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        equity_curve_file = os.path.join(output_dir, f"equity_curve_{portfolio_name}_{timestamp}.png")
        drawdown_file = os.path.join(output_dir, f"drawdowns_{portfolio_name}_{timestamp}.png")
        fig.savefig(equity_curve_file)
        plt.savefig(drawdown_file)
        plt.close('all')

        metrics = {
            "total_roi": total_roi,
            "max_drawdown": max_drawdown,
            "win_percentage": win_percentage,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "annualized_return": annual_return
        }
        return {"metrics": metrics, "files": {"equity_curve": equity_curve_file, "drawdowns": drawdown_file}}
    except Exception as e:
        logging.error(f"Error generating performance report: {e}")
        return {"metrics": {}, "files": {}}

# =============================================================================
# Fitness Evaluation Functions (Common)
# =============================================================================
def calculate_expectancy(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(np.abs(losses)) if len(losses) > 0 else 0
    return win_rate * avg_win - (1 - win_rate) * avg_loss

def evaluate_fitness(individual: List[int], daily_matrix: np.ndarray, metric: str) -> float:
    if not individual or not isinstance(individual, (list, tuple)):
        raise ValueError(f"Invalid individual: {individual}")
    if not isinstance(daily_matrix, np.ndarray) or daily_matrix.size == 0:
        raise ValueError("Invalid daily matrix")
    if max(individual) >= daily_matrix.shape[1]:
        raise IndexError(f"Individual index out of bounds: {max(individual)}")
    
    returns = daily_matrix[:, individual].mean(axis=1)
    m = metric.lower()
    
    if m == "roi":
        return np.sum(returns)
    elif m == "less max dd":
        eq = np.cumsum(returns)
        peak = np.maximum.accumulate(eq)
        max_dd = np.max(peak - eq)
        return -max_dd
    elif m == "ratio":
        roi = np.sum(returns)
        eq = np.cumsum(returns)
        peak = np.maximum.accumulate(eq)
        max_dd = np.max(peak - eq)
        if max_dd > 1e-6:
            fitness = roi / max_dd
        elif roi > 0:
            # Positive ROI with minimal drawdown is good
            fitness = roi * 100  # Multiply by a constant to give a high positive value
        elif roi < 0:
            # Negative ROI with minimal drawdown is bad
            fitness = roi * 10   # Still negative but not as extreme
        else:
            # ROI is zero, neutral case
            fitness = 0
        return fitness
    elif m == "win percentage":
        return np.sum(returns > 0) / len(returns)
    elif m == "profit factor":
        pos_sum = np.sum(returns[returns > 0])
        neg_sum = abs(np.sum(returns[returns < 0]))
        return pos_sum / neg_sum if neg_sum > 1e-6 else (1.0 if pos_sum > 0 else 0.0)
    elif m == "expectancy":
        return calculate_expectancy(returns)
    else:
        return np.sum(returns)

def compute_avg_pairwise_correlation(individual: List[int], corr_matrix: np.ndarray) -> float:
    correlations = []
    n = len(individual)
    for i in range(n):
        for j in range(i + 1, n):
            correlations.append(corr_matrix[individual[i], individual[j]])
    return np.mean(correlations) if correlations else 0

def evaluate_fitness_with_correlation(individual: List[int], daily_matrix: np.ndarray, metric: str,
                                      corr_matrix: np.ndarray, drawdown_threshold: float = 0) -> float:
    base_fitness = evaluate_fitness(individual, daily_matrix, metric)
    avg_corr = compute_avg_pairwise_correlation(individual, corr_matrix)
    penalty_weight = 10
    penalty = penalty_weight * avg_corr
    if metric.lower() in ["less max dd", "ratio"]:
        returns = daily_matrix[:, individual].mean(axis=1)
        eq = np.cumsum(returns)
        peak = np.maximum.accumulate(eq)
        max_dd = np.max(peak - eq)
        if max_dd < drawdown_threshold:
            penalty *= 0.5
    return base_fitness - penalty

def balanced_penalty(individual: List[int], file_labels: List[str], desired_ratio: Dict[str, int],
                     penalty_factor: float = 1.0) -> float:
    counts = Counter(file_labels[i] for i in individual)
    penalty = 0.0
    for key, desired in desired_ratio.items():
        diff = abs(counts.get(key, 0) - desired)
        penalty += diff * penalty_factor
    return penalty

def evaluate_fitness_with_balance(individual: List[int], daily_matrix: np.ndarray, metric: str,
                                  corr_matrix: np.ndarray, file_labels: List[str],
                                  desired_ratio: Dict[str, int], penalty_factor: float = 1.0,
                                  drawdown_threshold: float = 0) -> float:
    base = evaluate_fitness_with_correlation(individual, daily_matrix, metric, corr_matrix, drawdown_threshold)
    penalty = balanced_penalty(individual, file_labels, desired_ratio, penalty_factor)
    return base - penalty

def fitness_wrapper(ind: List[int], daily_matrix: np.ndarray, metric: str,
                    corr_matrix: np.ndarray, drawdown_threshold: float) -> float:
    if BALANCED_MODE:
        return evaluate_fitness_with_balance(ind, daily_matrix, metric, corr_matrix, FILE_LABELS, DESIRED_RATIO, PENALTY_FACTOR, drawdown_threshold)
    else:
        return evaluate_fitness_with_correlation(ind, daily_matrix, metric, corr_matrix, drawdown_threshold)

def fitness_wrapper_zone(ind: List[int], zone_matrix: np.ndarray, metric: str, corr_matrix: np.ndarray, drawdown_threshold: float, zone_weights: np.ndarray) -> float:
    return evaluate_fitness_zone(ind, zone_matrix, metric, corr_matrix, zone_weights, drawdown_threshold)

def parallel_fitness_eval(population: List[List[int]], daily_matrix: np.ndarray, metric: str,
                          corr_matrix: np.ndarray, drawdown_threshold: float = 0,
                          executor: Optional[ProcessPoolExecutor] = None) -> List[float]:
    # For Windows, use ThreadPoolExecutor instead of ProcessPoolExecutor for better compatibility
    if os.name == 'nt':
        logging.info("Windows detected - using ThreadPoolExecutor for better compatibility")
        from concurrent.futures import ThreadPoolExecutor
        
        max_workers = min(os.cpu_count() or 1, len(population), 8)
        
        try:
            if executor is None:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Process in chunks for better memory management
                    chunk_size = max(1, len(population) // max_workers)
                    futures = []
                    
                    for i in range(0, len(population), chunk_size):
                        chunk = population[i:i + chunk_size]
                        future = executor.submit(
                            _process_fitness_chunk_threaded, 
                            chunk, 
                            daily_matrix, 
                            metric, 
                            corr_matrix, 
                            drawdown_threshold
                        )
                        futures.append(future)
                    
                    fitnesses = []
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=300)  # 5 minute timeout per chunk
                            fitnesses.extend(result)
                        except Exception as e:
                            logging.error(f"Error in threaded fitness evaluation: {e}")
                            chunk_size_actual = len(population) // max_workers
                            fitnesses.extend([float('-inf')] * chunk_size_actual)
                    
                    return fitnesses
            else:
                # Use existing executor (should be ThreadPoolExecutor on Windows)
                chunk_size = max(1, len(population) // max_workers)
                futures = []
                
                for i in range(0, len(population), chunk_size):
                    chunk = population[i:i + chunk_size]
                    future = executor.submit(
                        _process_fitness_chunk_threaded, 
                        chunk, 
                        daily_matrix, 
                        metric, 
                        corr_matrix, 
                        drawdown_threshold
                    )
                    futures.append(future)
                
                fitnesses = []
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per chunk
                        fitnesses.extend(result)
                    except Exception as e:
                        logging.error(f"Error in threaded fitness evaluation: {e}")
                        chunk_size_actual = len(population) // max_workers
                        fitnesses.extend([float('-inf')] * chunk_size_actual)
                
                return fitnesses
        except Exception as e:
            logging.error(f"Error in threaded fitness evaluation, falling back to single-threaded: {e}")
            return [fitness_wrapper(ind, daily_matrix, metric, corr_matrix, drawdown_threshold) 
                    for ind in population]
    else:
        # For non-Windows systems, use ProcessPoolExecutor
        max_workers = min(os.cpu_count() or 1, len(population), 8)
        
        try:
            if executor is None:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    chunk_size = max(1, len(population) // max_workers)
                    futures = []
                    
                    for i in range(0, len(population), chunk_size):
                        chunk = population[i:i + chunk_size]
                        future = executor.submit(
                            _process_fitness_chunk, 
                            chunk, 
                            daily_matrix, 
                            metric, 
                            corr_matrix, 
                            drawdown_threshold
                        )
                        futures.append(future)
                    
                    fitnesses = []
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=300)
                            fitnesses.extend(result)
                        except Exception as e:
                            logging.error(f"Error in parallel fitness evaluation: {e}")
                            chunk_size_actual = len(population) // max_workers
                            fitnesses.extend([float('-inf')] * chunk_size_actual)
                    
                    return fitnesses
            else:
                chunk_size = max(1, len(population) // max_workers)
                futures = []
                
                for i in range(0, len(population), chunk_size):
                    chunk = population[i:i + chunk_size]
                    future = executor.submit(
                        _process_fitness_chunk, 
                        chunk, 
                        daily_matrix, 
                        metric, 
                        corr_matrix, 
                        drawdown_threshold
                    )
                    futures.append(future)
                
                fitnesses = []
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=300)
                        fitnesses.extend(result)
                    except Exception as e:
                        logging.error(f"Error in parallel fitness evaluation: {e}")
                        chunk_size_actual = len(population) // max_workers
                        fitnesses.extend([float('-inf')] * chunk_size_actual)
                
                return fitnesses
        except Exception as e:
            logging.error(f"Error in parallel fitness evaluation, falling back to single-threaded: {e}")
            return [fitness_wrapper(ind, daily_matrix, metric, corr_matrix, drawdown_threshold) 
                    for ind in population]

def _process_fitness_chunk_threaded(chunk: List[List[int]], daily_matrix: np.ndarray, metric: str,
                                   corr_matrix: np.ndarray, drawdown_threshold: float) -> List[float]:
    """Process a chunk of individuals using threading (safe for Windows)."""
    try:
        results = []
        for ind in chunk:
            try:
                result = fitness_wrapper(ind, daily_matrix, metric, corr_matrix, drawdown_threshold)
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing individual {ind}: {e}")
                results.append(float('-inf'))
        
        return results
    except Exception as e:
        logging.error(f"Error in threaded fitness chunk processing: {e}")
        return [float('-inf')] * len(chunk)

def _process_fitness_chunk_simple(chunk: List[List[int]]) -> List[float]:
    """Process a chunk of individuals using global data for better Windows compatibility."""
    global GLOBAL_DAILY_MATRIX, GLOBAL_CORR_MATRIX, GLOBAL_METRIC, GLOBAL_DRAWDOWN_THRESHOLD
    
    try:
        results = []
        for ind in chunk:
            try:
                result = fitness_wrapper(ind, GLOBAL_DAILY_MATRIX, GLOBAL_METRIC, 
                                       GLOBAL_CORR_MATRIX, GLOBAL_DRAWDOWN_THRESHOLD)
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing individual {ind}: {e}")
                results.append(float('-inf'))
        
        return results
    except Exception as e:
        logging.error(f"Error in fitness chunk processing: {e}")
        return [float('-inf')] * len(chunk)

def _process_fitness_chunk(chunk: List[List[int]], daily_matrix: np.ndarray, metric: str,
                          corr_matrix: np.ndarray, drawdown_threshold: float) -> List[float]:
    """Process a chunk of individuals for fitness evaluation."""
    try:
        # Convert numpy arrays to lists for better Windows compatibility
        daily_matrix_list = daily_matrix.tolist()
        corr_matrix_list = corr_matrix.tolist()
        
        results = []
        for ind in chunk:
            try:
                # Convert back to numpy arrays for processing
                daily_matrix_np = np.array(daily_matrix_list)
                corr_matrix_np = np.array(corr_matrix_list)
                result = fitness_wrapper(ind, daily_matrix_np, metric, corr_matrix_np, drawdown_threshold)
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing individual {ind}: {e}")
                results.append(float('-inf'))
        
        return results
    except Exception as e:
        logging.error(f"Error in fitness chunk processing: {e}")
        return [float('-inf')] * len(chunk)

# =============================================================================
# Zone-Based Fitness Functions
# =============================================================================
def evaluate_fitness_zone(individual: List[int], zone_matrix: np.ndarray, metric: str,
                          corr_matrix: np.ndarray, zone_weights: np.ndarray,
                          drawdown_threshold: float = 0) -> float:
    """
    Evaluate the fitness of a portfolio for zone-based optimization.
    """
    selected_returns = zone_matrix[:, :, individual]
    avg_returns = np.mean(selected_returns, axis=2)
    if random.random() < 0.005:
        logging.debug(f"Portfolio characteristics - Size: {len(individual)}, Strategies: {individual[:5]}...")
    num_zones = avg_returns.shape[1]
    if not isinstance(zone_weights, np.ndarray) or len(zone_weights) != num_zones:
        logging.warning(f"Zone weights length ({len(zone_weights) if isinstance(zone_weights, np.ndarray) else 'not array'}) does not match number of zones ({num_zones}). Using uniform weights.")
        zone_weights = np.ones(num_zones) / num_zones
    else:
        zone_weights = zone_weights / np.sum(zone_weights)
    weighted_returns = np.dot(avg_returns, zone_weights)
    if random.random() < 0.005:
        roi = np.sum(weighted_returns)
        eq = np.cumsum(weighted_returns)
        max_dd = np.max(np.maximum.accumulate(eq) - eq) if len(eq) > 0 else 0
        win_pct = np.sum(weighted_returns > 0) / len(weighted_returns) if len(weighted_returns) > 0 else 0
        logging.debug(f"Portfolio stats - ROI: {roi:.4f}, MaxDD: {max_dd:.4f}, Win%: {win_pct*100:.2f}%, Zones: {avg_returns.shape[1]}")
    
    # Only normalize if there's sufficient variance to avoid division by zero
    std_dev = np.std(weighted_returns)
    if std_dev > 1e-6:
        norm_returns = (weighted_returns - np.mean(weighted_returns)) / std_dev
    else:
        # If standard deviation is too small, use the raw returns
        norm_returns = weighted_returns
    
    m = metric.lower()
    if m == "roi":
        base_fitness = np.sum(weighted_returns)  # Use raw returns for ROI
    elif m == "less max dd":
        eq = np.cumsum(weighted_returns)  # Use raw returns for drawdown
        peak = np.maximum.accumulate(eq)
        max_dd = np.max(peak - eq) if len(eq) > 0 else 0
        base_fitness = -max_dd
    elif m == "ratio":
        # Use raw returns for ratio calculation
        roi = np.sum(weighted_returns)
        eq = np.cumsum(weighted_returns)
        peak = np.maximum.accumulate(eq)
        max_dd = np.max(peak - eq) if len(eq) > 0 else 0
        
        # Avoid division by zero and handle negative cases properly
        if max_dd > 1e-6:
            fitness = roi / max_dd
        elif roi > 0:
            # Positive ROI with minimal drawdown is good
            fitness = roi * 100  # Multiply by a constant to give a high positive value
        elif roi < 0:
            # Negative ROI with minimal drawdown is bad
            fitness = roi * 10   # Still negative but not as extreme
        else:
            # ROI is zero, neutral case
            fitness = 0
        
        return fitness
    elif m == "win percentage":
        base_fitness = np.sum(weighted_returns > 0) / len(weighted_returns) if len(weighted_returns) > 0 else 0
    elif m == "profit factor":
        pos_sum = np.sum(weighted_returns[weighted_returns > 0]) if any(weighted_returns > 0) else 0
        neg_sum = abs(np.sum(weighted_returns[weighted_returns < 0])) if any(weighted_returns < 0) else 0
        base_fitness = pos_sum / neg_sum if neg_sum > 1e-6 else (1.0 if pos_sum > 0 else 0.0)
    elif m == "expectancy":
        wins = weighted_returns[weighted_returns > 0]
        losses = weighted_returns[weighted_returns < 0]
        win_rate = len(wins) / len(weighted_returns) if len(weighted_returns) > 0 else 0
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(np.abs(losses)) if len(losses) > 0 else 0
        base_fitness = win_rate * avg_win - (1 - win_rate) * avg_loss
    else:
        base_fitness = np.sum(weighted_returns)
    
    # Add correlation penalty
    avg_corr = 0
    if len(individual) > 1:
        corrs = []
        for i in range(len(individual)):
            for j in range(i + 1, len(individual)):
                corrs.append(corr_matrix[individual[i], individual[j]])
        avg_corr = np.mean(corrs) if corrs else 0
    
    # Scale correlation penalty to be proportionate to base fitness
    penalty_weight = 0.5  # Reduced from previous value
    penalty = penalty_weight * max(0, avg_corr)  # Only penalize positive correlation
    
    # Apply drawdown threshold bonus if appropriate
    if m.lower() in ["less max dd", "ratio"]:
        eq = np.cumsum(weighted_returns)
        peak = np.maximum.accumulate(eq)
        max_dd = np.max(peak - eq) if len(eq) > 0 else 0
        if max_dd < drawdown_threshold:
            penalty *= 0.5  # Reduce penalty if drawdown is low
    
    return base_fitness - penalty
def tournament_selection(population: List[List[int]], fitnesses: List[float], tournament_size: int = 3) -> List[int]:
    """Select an individual using tournament selection."""
    selected_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
    tournament_fitnesses = [fitnesses[i] for i in selected_indices]
    winner_index = selected_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
    return population[winner_index].copy()

def crossover(parent1: List[int], parent2: List[int], total_strats: int, size: int) -> List[int]:
    """Create a child by combining genes from two parents."""
    child_pool = list(set(parent1 + parent2))
    if len(child_pool) < size:
        remaining = list(set(range(total_strats)) - set(child_pool))
        child_pool.extend(random.sample(remaining, min(size - len(child_pool), len(remaining))))
    return random.sample(child_pool, size)

def mutate(individual: List[int], total_strats: int, mutation_rate: float) -> List[int]:
    """Mutate an individual by randomly replacing strategies."""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            available = list(set(range(total_strats)) - set(mutated))
            if available:
                mutated[i] = random.choice(available)
    return mutated

def genetic_algorithm(daily_matrix: np.ndarray, n: int, metric: str, corr_matrix: np.ndarray,
                      output_dir: str, generations: int = 50, population_size: int = 30,
                      mutation_rate: float = 0.1, drawdown_threshold: float = 0,
                      executor: Optional[ProcessPoolExecutor] = None,
                      progress_callback: Optional[Callable[[float, str], bool]] = None) -> Tuple[List[int], float]:
    total_strats = daily_matrix.shape[1]
    checkpoint_state = load_checkpoint("ga", output_dir, n) if USE_CHECKPOINT else {}
    if checkpoint_state and "population" in checkpoint_state:
        population = checkpoint_state["population"]
        best_individual = checkpoint_state["best_individual"]
        best_fitness = checkpoint_state["best_fitness"]
        start_gen = checkpoint_state["generation"] + 1
        logging.info(f"Resuming GA from generation {start_gen} for portfolio size {n}")
    else:
        population = [random.sample(range(total_strats), n) for _ in range(population_size)]
        best_individual = None
        best_fitness = -np.inf
        start_gen = 0
    for gen in tqdm(range(start_gen, generations), desc="GA Optimization"):
        if executor is None:
            with ProcessPoolExecutor() as temp_executor:
                fitnesses = list(temp_executor.map(
                    fitness_wrapper,
                    population,
                    [daily_matrix] * len(population),
                    [metric] * len(population),
                    [corr_matrix] * len(population),
                    [drawdown_threshold] * len(population)
                ))
        else:
            fitnesses = list(executor.map(
                fitness_wrapper,
                population,
                [daily_matrix] * len(population),
                [metric] * len(population),
                [corr_matrix] * len(population),
                [drawdown_threshold] * len(population)
            ))
        max_fit_idx = fitnesses.index(max(fitnesses))
        if fitnesses[max_fit_idx] > best_fitness:
            best_fitness = fitnesses[max_fit_idx]
            best_individual = population[max_fit_idx].copy()
        new_population = []
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2, total_strats, n)
            child = mutate(child, total_strats, mutation_rate)
            new_population.append(child)
        population = new_population
        logging.info(f"[GA] Gen {gen+1}/{generations}, Best Fitness = {best_fitness:.4f}")
        if USE_CHECKPOINT:
            state = {
                "population": population, 
                "best_individual": best_individual,
                "best_fitness": best_fitness, 
                "generation": gen
            }
            save_checkpoint(state, "ga", output_dir, n)
        if progress_callback and not progress_callback((gen+1)/generations*100,
                                                       f"[GA] Gen {gen+1}/{generations}, Best Fitness = {best_fitness:.4f}"):
            logging.info("GA optimization stopped by user request")
            break
    return best_individual, best_fitness

def pso_algorithm(daily_matrix: np.ndarray, n: int, metric: str, corr_matrix: np.ndarray,
                  output_dir: str, iterations: int = 50, swarm_size: int = 30,
                  drawdown_threshold: float = 0, executor: Optional[ProcessPoolExecutor] = None,
                  progress_callback: Optional[Callable[[float, str], bool]] = None) -> Tuple[List[int], float]:
    total_strats = daily_matrix.shape[1]
    checkpoint_state = load_checkpoint("pso", output_dir, n) if USE_CHECKPOINT else {}
    if checkpoint_state and "swarm" in checkpoint_state:
        swarm = checkpoint_state["swarm"]
        best_particle = checkpoint_state["best_particle"]
        best_fitness = checkpoint_state["best_fitness"]
        start_iter = checkpoint_state["iteration"] + 1
        logging.info(f"Resuming PSO from iteration {start_iter} for portfolio size {n}")
    else:
        swarm = [random.sample(range(total_strats), n) for _ in range(swarm_size)]
        best_particle = swarm[0] if swarm else None
        best_fitness = -np.inf
        start_iter = 0
    for it in tqdm(range(start_iter, iterations), desc="PSO Optimization"):
        if executor is None:
            with ProcessPoolExecutor() as temp_executor:
                fitnesses = list(temp_executor.map(
                    fitness_wrapper,
                    swarm,
                    [daily_matrix] * len(swarm),
                    [metric] * len(swarm),
                    [corr_matrix] * len(swarm),
                    [drawdown_threshold] * len(swarm)
                ))
        else:
            fitnesses = list(executor.map(
                fitness_wrapper,
                swarm,
                [daily_matrix] * len(swarm),
                [metric] * len(swarm),
                [corr_matrix] * len(swarm),
                [drawdown_threshold] * len(swarm)
            ))
        for idx, fit in enumerate(fitnesses):
            if fit > best_fitness:
                best_fitness = fit
                best_particle = swarm[idx].copy()
        new_swarm = []
        for particle in swarm:
            new_particle = mutate(particle, total_strats, mutation_rate=0.05)
            new_swarm.append(new_particle)
        swarm = new_swarm
        logging.info(f"[PSO] Iter {it+1}/{iterations}, Best Fitness = {best_fitness:.4f}")
        if USE_CHECKPOINT:
            state = {
                "swarm": swarm, 
                "best_particle": best_particle,
                "best_fitness": best_fitness, 
                "iteration": it
            }
            save_checkpoint(state, "pso", output_dir, n)
        if progress_callback and not progress_callback((it+1)/iterations*100,
                                                       f"[PSO] Iter {it+1}/{iterations}, Best Fitness = {best_fitness:.4f}"):
            logging.info("PSO optimization stopped by user request")
            break
    return best_particle, best_fitness

def simulated_annealing(daily_matrix: np.ndarray, n: int, metric: str, corr_matrix: np.ndarray,
                        output_dir: str, iterations: int = 1000, drawdown_threshold: float = 0,
                        progress_callback: Optional[Callable[[float, str], bool]] = None) -> Tuple[List[int], float]:
    total_strats = daily_matrix.shape[1]
    best_solution = random.sample(range(total_strats), n)
    best_fitness = fitness_wrapper(best_solution, daily_matrix, metric, corr_matrix, drawdown_threshold)
    for it in tqdm(range(iterations), desc="SA Optimization"):
        candidate = best_solution.copy()
        idx = random.randrange(n)
        candidate[idx] = random.choice(list(set(range(total_strats)) - set(candidate)))
        candidate_fitness = fitness_wrapper(candidate, daily_matrix, metric, corr_matrix, drawdown_threshold)
        if candidate_fitness > best_fitness:
            best_solution = candidate
            best_fitness = candidate_fitness
        if progress_callback and it % 100 == 0:
            progress_callback((it+1)/iterations*100, f"[SA] Iter {it+1}, Best Fitness = {best_fitness:.4f}")
    return best_solution, best_fitness

def differential_evolution(daily_matrix: np.ndarray, n: int, metric: str, corr_matrix: np.ndarray,
                           output_dir: str, population_size: int = 30, iterations: int = 50,
                           drawdown_threshold: float = 0,
                           progress_callback: Optional[Callable[[float, str], bool]] = None) -> Tuple[List[int], float]:
    total_strats = daily_matrix.shape[1]
    population = [random.sample(range(total_strats), n) for _ in range(population_size)]
    best_solution = population[0]
    best_fitness = fitness_wrapper(best_solution, daily_matrix, metric, corr_matrix, drawdown_threshold)
    for it in tqdm(range(iterations), desc="DE Optimization"):
        new_population = []
        for individual in population:
            partner = random.choice(population)
            child = [individual[i] if random.random() < 0.5 else partner[i] for i in range(n)]
            child = list(set(child))
            if len(child) < n:
                remaining = list(set(range(total_strats)) - set(child))
                child += random.sample(remaining, n - len(child))
            child_fitness = fitness_wrapper(child, daily_matrix, metric, corr_matrix, drawdown_threshold)
            new_population.append(child)
            if child_fitness > best_fitness:
                best_solution = child
                best_fitness = child_fitness
        population = new_population
        if progress_callback and it % 10 == 0:
            progress_callback((it+1)/iterations*100, f"[DE] Iter {it+1}, Best Fitness = {best_fitness:.4f}")
    return best_solution, best_fitness

def ant_colony_optimization(daily_matrix: np.ndarray, n: int, metric: str, corr_matrix: np.ndarray,
                            output_dir: str, iterations: int = 50, drawdown_threshold: float = 0,
                            progress_callback: Optional[Callable[[float, str], bool]] = None) -> Tuple[List[int], float]:
    total_strats = daily_matrix.shape[1]
    best_solution = random.sample(range(total_strats), n)
    best_fitness = fitness_wrapper(best_solution, daily_matrix, metric, corr_matrix, drawdown_threshold)
    for it in tqdm(range(iterations), desc="ACO Optimization"):
        candidate = random.sample(range(total_strats), n)
        candidate_fitness = fitness_wrapper(candidate, daily_matrix, metric, corr_matrix, drawdown_threshold)
        if candidate_fitness > best_fitness:
            best_solution = candidate
            best_fitness = candidate_fitness
        if progress_callback and it % 10 == 0:
            progress_callback((it+1)/iterations*100, f"[ACO] Iter {it+1}, Best Fitness = {best_fitness:.4f}")
    return best_solution, best_fitness

def hill_climbing(daily_matrix: np.ndarray, n: int, metric: str, corr_matrix: np.ndarray,
                  output_dir: str, iterations: int = 200, drawdown_threshold: float = 0,
                  progress_callback: Optional[Callable[[float, str], bool]] = None) -> Tuple[List[int], float]:
    total_strats = daily_matrix.shape[1]
    current_solution = random.sample(range(total_strats), n)
    current_fitness = fitness_wrapper(current_solution, daily_matrix, metric, corr_matrix, drawdown_threshold)
    for it in tqdm(range(iterations), desc="Hill Climbing"):
        neighbor = current_solution.copy()
        idx = random.randrange(n)
        neighbor[idx] = random.choice(list(set(range(total_strats)) - set(neighbor)))
        neighbor_fitness = fitness_wrapper(neighbor, daily_matrix, metric, corr_matrix, drawdown_threshold)
        if neighbor_fitness > current_fitness:
            current_solution = neighbor
            current_fitness = neighbor_fitness
        if progress_callback and it % 10 == 0:
            progress_callback((it+1)/iterations*100, f"[HC] Iter {it+1}, Best Fitness = {current_fitness:.4f}")
    return current_solution, current_fitness

def bayesian_optimization(daily_matrix: np.ndarray, n: int, metric: str, corr_matrix: np.ndarray,
                          output_dir: str, iterations: int = 50, drawdown_threshold: float = 0,
                          progress_callback: Optional[Callable[[float, str], bool]] = None) -> Tuple[List[int], float]:
    total_strats = daily_matrix.shape[1]
    best_solution = random.sample(range(total_strats), n)
    best_fitness = fitness_wrapper(best_solution, daily_matrix, metric, corr_matrix, drawdown_threshold)
    for it in tqdm(range(iterations), desc="Bayesian Optimization"):
        candidate = random.sample(range(total_strats), n)
        candidate_fitness = fitness_wrapper(candidate, daily_matrix, metric, corr_matrix, drawdown_threshold)
        if candidate_fitness > best_fitness:
            best_solution = candidate
            best_fitness = candidate_fitness
        if progress_callback and it % 10 == 0:
            progress_callback((it+1)/iterations*100, f"[BO] Iter {it+1}, Best Fitness = {best_fitness:.4f}")
    return best_solution, best_fitness

# =============================================================================
# Zone-Based Mode Optimization Algorithms
# =============================================================================
def genetic_algorithm_zone(zone_matrix: np.ndarray, n: int, metric: str, corr_matrix: np.ndarray,
                           output_dir: str, zone_weights: np.ndarray, generations: int = 50,
                           population_size: int = 30, mutation_rate: float = 0.1,
                           drawdown_threshold: float = 0, executor: Optional[ProcessPoolExecutor] = None,
                           progress_callback: Optional[Callable[[float, str], bool]] = None) -> Tuple[List[int], float]:
    total_strats = zone_matrix.shape[2]
    checkpoint_state = load_checkpoint("ga_zone", output_dir, n) if USE_CHECKPOINT else {}
    if checkpoint_state and "population" in checkpoint_state:
        population = checkpoint_state["population"]
        best_individual = checkpoint_state["best_individual"]
        best_fitness = checkpoint_state["best_fitness"]
        start_gen = checkpoint_state["generation"] + 1
        logging.info(f"Resuming GA (Zone) from generation {start_gen} for portfolio size {n}")
    else:
        population = [random.sample(range(total_strats), n) for _ in range(population_size)]
        best_individual = None
        best_fitness = -np.inf
        start_gen = 0
    for gen in tqdm(range(start_gen, generations), desc="GA Optimization (Zone)"):
        if executor is None:
            with ProcessPoolExecutor() as temp_executor:
                fitnesses = list(temp_executor.map(
                    fitness_wrapper_zone,
                    population,
                    [zone_matrix] * len(population),
                    [metric] * len(population),
                    [corr_matrix] * len(population),
                    [drawdown_threshold] * len(population),
                    [zone_weights] * len(population)
                ))
        else:
            fitnesses = list(executor.map(
                fitness_wrapper_zone,
                population,
                [zone_matrix] * len(population),
                [metric] * len(population),
                [corr_matrix] * len(population),
                [drawdown_threshold] * len(population),
                [zone_weights] * len(population)
            ))
        max_fit_idx = fitnesses.index(max(fitnesses))
        if fitnesses[max_fit_idx] > best_fitness:
            best_fitness = fitnesses[max_fit_idx]
            best_individual = population[max_fit_idx].copy()
        new_population = []
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2, total_strats, n)
            child = mutate(child, total_strats, mutation_rate)
            new_population.append(child)
        population = new_population
        logging.info(f"[GA-Zone] Gen {gen+1}/{generations}, Best Fitness = {best_fitness:.4f}")
        if USE_CHECKPOINT:
            state = {
                "population": population, 
                "best_individual": best_individual,
                "best_fitness": best_fitness, 
                "generation": gen
            }
            save_checkpoint(state, "ga_zone", output_dir, n)
        if progress_callback and not progress_callback((gen+1)/generations*100,
                                                       f"[GA-Zone] Gen {gen+1}/{generations}, Best Fitness = {best_fitness:.4f}"):
            logging.info("GA-Zone optimization stopped by user request")
            break
    return best_individual, best_fitness

def pso_algorithm_zone(zone_matrix: np.ndarray, n: int, metric: str, corr_matrix: np.ndarray,
                       output_dir: str, iterations: int = 50, swarm_size: int = 30,
                       zone_weights: np.ndarray = None, drawdown_threshold: float = 0,
                       executor: Optional[ProcessPoolExecutor] = None,
                       progress_callback: Optional[Callable[[float, str], bool]] = None) -> Tuple[List[int], float]:
    total_strats = zone_matrix.shape[2]
    checkpoint_state = load_checkpoint("pso_zone", output_dir, n) if USE_CHECKPOINT else {}
    if checkpoint_state and "swarm" in checkpoint_state:
        swarm = checkpoint_state["swarm"]
        best_particle = checkpoint_state["best_particle"]
        best_fitness = checkpoint_state["best_fitness"]
        start_iter = checkpoint_state["iteration"] + 1
        logging.info(f"Resuming PSO (Zone) from iteration {start_iter} for portfolio size {n}")
    else:
        swarm = [random.sample(range(total_strats), n) for _ in range(swarm_size)]
        best_particle = swarm[0] if swarm else None
        best_fitness = -np.inf
        start_iter = 0
    for it in tqdm(range(start_iter, iterations), desc="PSO Optimization (Zone)"):
        if executor is None:
            with ProcessPoolExecutor() as temp_executor:
                fitnesses = list(temp_executor.map(
                    fitness_wrapper_zone,
                    swarm,
                    [zone_matrix] * len(swarm),
                    [metric] * len(swarm),
                    [corr_matrix] * len(swarm),
                    [drawdown_threshold] * len(swarm),
                    [zone_weights] * len(swarm)
                ))
        else:
            fitnesses = list(executor.map(
                fitness_wrapper_zone,
                swarm,
                [zone_matrix] * len(swarm),
                [metric] * len(swarm),
                [corr_matrix] * len(swarm),
                [drawdown_threshold] * len(swarm),
                [zone_weights] * len(swarm)
            ))
        for idx, fit in enumerate(fitnesses):
            if fit > best_fitness:
                best_fitness = fit
                best_particle = swarm[idx].copy()
        new_swarm = []
        for particle in swarm:
            new_particle = mutate(particle, total_strats, mutation_rate=0.05)
            new_swarm.append(new_particle)
        swarm = new_swarm
        logging.info(f"[PSO-Zone] Iter {it+1}/{iterations}, Best Fitness = {best_fitness:.4f}")
        if USE_CHECKPOINT:
            state = {
                "swarm": swarm, 
                "best_particle": best_particle,
                "best_fitness": best_fitness, 
                "iteration": it
            }
            save_checkpoint(state, "pso_zone", output_dir, n)
        if progress_callback and not progress_callback((it+1)/iterations*100,
                                                       f"[PSO-Zone] Iter {it+1}/{iterations}, Best Fitness = {best_fitness:.4f}"):
            logging.info("PSO-Zone optimization stopped by user request")
            break
    return best_particle, best_fitness

def simulated_annealing_zone(zone_matrix: np.ndarray, n: int, metric: str, corr_matrix: np.ndarray,
                             output_dir: str, zone_weights: np.ndarray, iterations: int = 1000,
                             drawdown_threshold: float = 0,
                             progress_callback: Optional[Callable[[float, str], bool]] = None) -> Tuple[List[int], float]:
    total_strats = zone_matrix.shape[2]
    best_solution = random.sample(range(total_strats), n)
    best_fitness = fitness_wrapper_zone(best_solution, zone_matrix, metric, corr_matrix, drawdown_threshold, zone_weights)
    for it in tqdm(range(iterations), desc="SA Optimization (Zone)"):
        candidate = best_solution.copy()
        idx = random.randrange(n)
        candidate[idx] = random.choice(list(set(range(total_strats)) - set(candidate)))
        candidate_fitness = fitness_wrapper_zone(candidate, zone_matrix, metric, corr_matrix, drawdown_threshold, zone_weights)
        if candidate_fitness > best_fitness:
            best_solution = candidate
            best_fitness = candidate_fitness
        if progress_callback and it % 100 == 0:
            progress_callback((it+1)/iterations*100, f"[SA-Zone] Iter {it+1}, Best Fitness = {best_fitness:.4f}")
    return best_solution, best_fitness

def differential_evolution_zone(zone_matrix: np.ndarray, n: int, metric: str, corr_matrix: np.ndarray,
                                output_dir: str, zone_weights: np.ndarray, population_size: int = 30,
                                iterations: int = 50, drawdown_threshold: float = 0,
                                progress_callback: Optional[Callable[[float, str], bool]] = None) -> Tuple[List[int], float]:
    total_strats = zone_matrix.shape[2]
    population = [random.sample(range(total_strats), n) for _ in range(population_size)]
    best_solution = population[0]
    best_fitness = fitness_wrapper_zone(best_solution, zone_matrix, metric, corr_matrix, drawdown_threshold, zone_weights)
    for it in tqdm(range(iterations), desc="DE Optimization (Zone)"):
        new_population = []
        for individual in population:
            partner = random.choice(population)
            child = [individual[i] if random.random() < 0.5 else partner[i] for i in range(n)]
            child = list(set(child))
            if len(child) < n:
                remaining = list(set(range(total_strats)) - set(child))
                child += random.sample(remaining, n - len(child))
            child_fitness = fitness_wrapper_zone(child, zone_matrix, metric, corr_matrix, drawdown_threshold, zone_weights)
            new_population.append(child)
            if child_fitness > best_fitness:
                best_solution = child
                best_fitness = child_fitness
        population = new_population
        if progress_callback and it % 10 == 0:
            progress_callback((it+1)/iterations*100, f"[DE-Zone] Iter {it+1}, Best Fitness = {best_fitness:.4f}")
    return best_solution, best_fitness

def ant_colony_optimization_zone(zone_matrix: np.ndarray, n: int, metric: str, corr_matrix: np.ndarray,
                                 output_dir: str, zone_weights: np.ndarray, iterations: int = 50,
                                 drawdown_threshold: float = 0,
                                 progress_callback: Optional[Callable[[float, str], bool]] = None) -> Tuple[List[int], float]:
    total_strats = zone_matrix.shape[2]
    best_solution = random.sample(range(total_strats), n)
    best_fitness = fitness_wrapper_zone(best_solution, zone_matrix, metric, corr_matrix, drawdown_threshold, zone_weights)
    for it in tqdm(range(iterations), desc="ACO Optimization (Zone)"):
        candidate = random.sample(range(total_strats), n)
        candidate_fitness = fitness_wrapper_zone(candidate, zone_matrix, metric, corr_matrix, drawdown_threshold, zone_weights)
        if candidate_fitness > best_fitness:
            best_solution = candidate
            best_fitness = candidate_fitness
        if progress_callback and it % 10 == 0:
            progress_callback((it+1)/iterations*100, f"[ACO-Zone] Iter {it+1}, Best Fitness = {best_fitness:.4f}")
    return best_solution, best_fitness

def hill_climbing_zone(zone_matrix: np.ndarray, n: int, metric: str, corr_matrix: np.ndarray,
                       output_dir: str, zone_weights: np.ndarray, iterations: int = 200,
                       drawdown_threshold: float = 0,
                       progress_callback: Optional[Callable[[float, str], bool]] = None) -> Tuple[List[int], float]:
    total_strats = zone_matrix.shape[2]
    current_solution = random.sample(range(total_strats), n)
    current_fitness = fitness_wrapper_zone(current_solution, zone_matrix, metric, corr_matrix, drawdown_threshold, zone_weights)
    for it in tqdm(range(iterations), desc="Hill Climbing (Zone)"):
        neighbor = current_solution.copy()
        idx = random.randrange(n)
        neighbor[idx] = random.choice(list(set(range(total_strats)) - set(neighbor)))
        neighbor_fitness = fitness_wrapper_zone(neighbor, zone_matrix, metric, corr_matrix, drawdown_threshold, zone_weights)
        if neighbor_fitness > current_fitness:
            current_solution = neighbor
            current_fitness = neighbor_fitness
        if progress_callback and it % 10 == 0:
            progress_callback((it+1)/iterations*100, f"[HC-Zone] Iter {it+1}, Best Fitness = {current_fitness:.4f}")
    return current_solution, current_fitness

def bayesian_optimization_zone(zone_matrix: np.ndarray, n: int, metric: str, corr_matrix: np.ndarray,
                               output_dir: str, zone_weights: np.ndarray, iterations: int = 50,
                               drawdown_threshold: float = 0,
                               progress_callback: Optional[Callable[[float, str], bool]] = None) -> Tuple[List[int], float]:
    total_strats = zone_matrix.shape[2]
    best_solution = random.sample(range(total_strats), n)
    best_fitness = fitness_wrapper_zone(best_solution, zone_matrix, metric, corr_matrix, drawdown_threshold, zone_weights)
    for it in tqdm(range(iterations), desc="Bayesian Optimization (Zone)"):
        candidate = random.sample(range(total_strats), n)
        candidate_fitness = fitness_wrapper_zone(candidate, zone_matrix, metric, corr_matrix, drawdown_threshold, zone_weights)
        if candidate_fitness > best_fitness:
            best_solution = candidate
            best_fitness = candidate_fitness
        if progress_callback and it % 10 == 0:
            progress_callback((it+1)/iterations*100, f"[BO-Zone] Iter {it+1}, Best Fitness = {best_fitness:.4f}")
    return best_solution, best_fitness

# =============================================================================
# Parallel Evaluation Module for Individual Strategy Performance
# =============================================================================
def compute_strategy_metrics(series: pd.Series) -> Dict[str, float]:
    returns = pd.to_numeric(series, errors="coerce").fillna(0).values
    roi = np.sum(returns)
    cum_returns = np.cumsum(returns)
    peak = np.maximum.accumulate(cum_returns)
    max_drawdown = np.max(peak - cum_returns)
    win_percentage = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
    pos_returns = returns[returns > 0]
    neg_returns = returns[returns < 0]
    avg_win = np.mean(pos_returns) if len(pos_returns) > 0 else 0
    avg_loss = np.mean(np.abs(neg_returns)) if len(neg_returns) > 0 else 0
    profit_factor = np.sum(pos_returns) / np.sum(np.abs(neg_returns)) if np.sum(np.abs(neg_returns)) != 0 else np.inf
    expectancy = (win_percentage * avg_win) - ((1 - win_percentage) * avg_loss)
    return {
        "ROI": roi,
        "Max Drawdown": max_drawdown,
        "Win Percentage": win_percentage,
        "Profit Factor": profit_factor,
        "Expectancy": expectancy
    }

def evaluate_individual_strategies_parallel(data: pd.DataFrame) -> pd.DataFrame:
    strategy_names = data.columns[3:]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(compute_strategy_metrics, [data[strat] for strat in strategy_names]))
    metrics_df = pd.DataFrame(results, index=strategy_names)
    return metrics_df

# =============================================================================
# Secondary Zone-Specific Optimization Function
# =============================================================================
def run_zone_specific_optimization_for_all_zones(consolidated_df, run_dir, config, progress_callback=None):
    """
    Run zone-specific optimization for each zone independently.
    Fixed version with proper result handling for each zone.
    """
    zone_results = {}
    desired_zone_ids = {"zone1", "zone2", "zone3", "zone4"}
    unique_zones = consolidated_df["Zone"].unique()
    use_ga = str(config['ALGORITHMS'].get('use_genetic_algorithm', "True")).lower() in ["true", "1", "yes"]
    use_pso = str(config['ALGORITHMS'].get('use_particle_swarm', "True")).lower() in ["true", "1", "yes"]
    use_sa = str(config['ALGORITHMS'].get('use_simulated_annealing', "True")).lower() in ["true", "1", "yes"]
    use_de = str(config['ALGORITHMS'].get('use_differential_evolution', "True")).lower() in ["true", "1", "yes"]
    use_aco = str(config['ALGORITHMS'].get('use_ant_colony', "True")).lower() in ["true", "1", "yes"]
    use_hc = str(config['ALGORITHMS'].get('use_hill_climbing', "True")).lower() in ["true", "1", "yes"]
    use_bo = str(config['ALGORITHMS'].get('use_bayesian_optimization', "True")).lower() in ["true", "1", "yes"]
    if not any([use_ga, use_pso, use_sa, use_de, use_aco, use_hc, use_bo]):
        logging.warning("No optimization algorithms selected for zone-specific optimization. Defaulting to GA.")
        use_ga = True
    metric = config['GENERAL'].get('metric', 'ratio').lower()
    ga_generations = max(10, int(config['GENERAL'].get('ga_generations', 50)))
    drawdown_threshold = float(config['OPTIMIZATION'].get('drawdown_threshold', 0))
    if "ZONE_SPECIFIC_OPTIMIZATION" not in config:
        logging.info("Creating ZONE_SPECIFIC_OPTIMIZATION section in config")
        config["ZONE_SPECIFIC_OPTIMIZATION"] = {
            "enable": "True",
            "min_size": config['PORTFOLIO'].get('min_size', "3"),
            "max_size": config['PORTFOLIO'].get('max_size', "5"),
            "population_size": config['PORTFOLIO'].get('population_size', "30"),
            "mutation_rate": config['PORTFOLIO'].get('mutation_rate', "0.1")
        }
    try:
        z_min_size = int(config['ZONE_SPECIFIC_OPTIMIZATION'].get("min_size", config['PORTFOLIO']['min_size'])) if 'ZONE_SPECIFIC_OPTIMIZATION' in config else int(config['PORTFOLIO']['min_size'])
    except (ValueError, TypeError, KeyError):
        logging.warning("Invalid zone-specific min_size, using portfolio min_size instead")
        try:
            z_min_size = int(config['PORTFOLIO']['min_size'])
        except (ValueError, TypeError, KeyError):
            logging.warning("Invalid portfolio min_size, using default value of 3")
            z_min_size = 3
    try:
        z_max_size = int(config['ZONE_SPECIFIC_OPTIMIZATION'].get("max_size", config['PORTFOLIO']['max_size'])) if 'ZONE_SPECIFIC_OPTIMIZATION' in config else int(config['PORTFOLIO']['max_size'])
    except (ValueError, TypeError, KeyError):
        logging.warning("Invalid zone-specific max_size, using portfolio max_size instead")
        try:
            z_max_size = int(config['PORTFOLIO']['max_size'])
        except (ValueError, TypeError, KeyError):
            logging.warning("Invalid portfolio max_size, using default value of 5")
            z_max_size = 5
    try:
        z_population_size = int(config['ZONE_SPECIFIC_OPTIMIZATION'].get("population_size", config['PORTFOLIO']['population_size'])) if 'ZONE_SPECIFIC_OPTIMIZATION' in config else int(config['PORTFOLIO']['population_size'])
    except (ValueError, TypeError, KeyError):
        logging.warning("Invalid zone-specific population_size, using default value of 30")
        z_population_size = 30
    try:
        z_mutation_rate = float(config['ZONE_SPECIFIC_OPTIMIZATION'].get("mutation_rate", config['PORTFOLIO']['mutation_rate'])) if 'ZONE_SPECIFIC_OPTIMIZATION' in config else float(config['PORTFOLIO']['mutation_rate'])
    except (ValueError, TypeError, KeyError):
        logging.warning("Invalid zone-specific mutation_rate, using default value of 0.1")
        z_mutation_rate = 0.1
    if z_min_size > z_max_size:
        logging.warning(f"Zone-specific min_size ({z_min_size}) > max_size ({z_max_size}). Setting min_size = max_size")
        z_min_size = z_max_size
    for zone in unique_zones:
        zone_id = zone.lower().replace(" ", "")
        if zone_id not in desired_zone_ids:
            continue
        logging.info(f"Running zone-specific optimization for zone: {zone}")
        if progress_callback:
            progress_callback(0, f"Optimizing zone: {zone}")
        zone_data = consolidated_df[consolidated_df["Zone"] == zone].copy()
        if str(config['OPTIMIZATION'].get('apply_ulta_logic', "True")).lower() in ["true", "1", "yes"]:
            zone_data, _ = apply_ulta_logic(zone_data)
        if zone_data.shape[1] > 3:
            strat_columns = zone_data.columns[3:]
            try:
                daily_matrix = zone_data[strat_columns].to_numpy().astype(float)
            except Exception as e:
                logging.error(f"Error converting zone data to numeric for zone {zone}: {e}")
                num_data = []
                for col in strat_columns:
                    try:
                        num_data.append(pd.to_numeric(zone_data[col], errors='coerce').fillna(0).values)
                    except Exception as e2:
                        num_data.append(np.zeros(len(zone_data)))
                daily_matrix = np.column_stack(num_data)
        else:
            logging.error(f"Zone {zone} does not contain enough data columns.")
            continue
        try:
            corr_matrix = np.corrcoef(daily_matrix.T)
            corr_matrix = np.nan_to_num(corr_matrix)
        except Exception as e:
            logging.error(f"Error computing correlation for zone {zone}: {e}")
            corr_matrix = np.eye(daily_matrix.shape[1])
        max_possible_size = daily_matrix.shape[1]
        if z_min_size > max_possible_size:
            logging.warning(f"Reducing min portfolio size for zone {zone} from {z_min_size} to {max_possible_size} due to data limitations")
            z_min_size = max_possible_size
        if z_max_size > max_possible_size:
            logging.warning(f"Reducing max portfolio size for zone {zone} from {z_max_size} to {max_possible_size} due to data limitations")
            z_max_size = max_possible_size
        logging.info(f"Zone {zone}: Using portfolio size range min={z_min_size}, max={z_max_size}")
        best_zone_fitness = -np.inf
        best_zone_method = None
        best_zone_size = None
        best_zone_solution = None
        zone_strat_names = list(strat_columns)
        zone_overall_results = []
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            for size in range(z_min_size, z_max_size + 1):
                zone_portfolio_results = []
                msg = f"Zone {zone}: Optimizing for portfolio size {size}"
                logging.info(msg)
                if progress_callback:
                    progress_callback(0, msg)
                if size > daily_matrix.shape[1]:
                    logging.warning(f"Portfolio size {size} exceeds available strategies {daily_matrix.shape[1]} for zone {zone}")
                    continue
                if use_ga:
                    try:
                        sol, fit = genetic_algorithm(daily_matrix, size, metric, corr_matrix,
                                                     run_dir, generations=ga_generations,
                                                     population_size=z_population_size, mutation_rate=z_mutation_rate,
                                                     executor=executor, progress_callback=progress_callback)
                        if isinstance(sol, list) and len(sol) == size:
                            zone_portfolio_results.append(("GA", sol, fit))
                    except Exception as e:
                        logging.error(f"Error running GA for zone {zone}, size {size}: {e}")
                if use_pso:
                    try:
                        sol, fit = pso_algorithm(daily_matrix, size, metric, corr_matrix,
                                                 run_dir, iterations=ga_generations, swarm_size=z_population_size,
                                                 drawdown_threshold=drawdown_threshold, executor=executor,
                                                 progress_callback=progress_callback)
                        if isinstance(sol, list) and len(sol) == size:
                            zone_portfolio_results.append(("PSO", sol, fit))
                    except Exception as e:
                        logging.error(f"Error running PSO for zone {zone}, size {size}: {e}")
                if use_sa:
                    try:
                        sol, fit = simulated_annealing(daily_matrix, size, metric, corr_matrix,
                                                       run_dir, iterations=1000, drawdown_threshold=drawdown_threshold,
                                                       progress_callback=progress_callback)
                        if isinstance(sol, list) and len(sol) == size:
                            zone_portfolio_results.append(("SA", sol, fit))
                    except Exception as e:
                        logging.error(f"Error running SA for zone {zone}, size {size}: {e}")
                if use_de:
                    try:
                        sol, fit = differential_evolution(daily_matrix, size, metric, corr_matrix,
                                                          run_dir, population_size=z_population_size, iterations=ga_generations,
                                                          drawdown_threshold=drawdown_threshold, progress_callback=progress_callback)
                        if isinstance(sol, list) and len(sol) == size:
                            zone_portfolio_results.append(("DE", sol, fit))
                    except Exception as e:
                        logging.error(f"Error running DE for zone {zone}, size {size}: {e}")
                if use_aco:
                    try:
                        sol, fit = ant_colony_optimization(daily_matrix, size, metric, corr_matrix,
                                                           run_dir, iterations=ga_generations, drawdown_threshold=drawdown_threshold,
                                                           progress_callback=progress_callback)
                        if isinstance(sol, list) and len(sol) == size:
                            zone_portfolio_results.append(("ACO", sol, fit))
                    except Exception as e:
                        logging.error(f"Error running ACO for zone {zone}, size {size}: {e}")
                if use_hc:
                    try:
                        sol, fit = hill_climbing(daily_matrix, size, metric, corr_matrix,
                                                 run_dir, iterations=200, drawdown_threshold=drawdown_threshold,
                                                 progress_callback=progress_callback)
                        if isinstance(sol, list) and len(sol) == size:
                            zone_portfolio_results.append(("HC", sol, fit))
                    except Exception as e:
                        logging.error(f"Error running HC for zone {zone}, size {size}: {e}")
                if use_bo:
                    try:
                        sol, fit = bayesian_optimization(daily_matrix, size, metric, corr_matrix,
                                                         run_dir, iterations=ga_generations, drawdown_threshold=drawdown_threshold,
                                                         progress_callback=progress_callback)
                        if isinstance(sol, list) and len(sol) == size:
                            zone_portfolio_results.append(("BO", sol, fit))
                    except Exception as e:
                        logging.error(f"Error running BO for zone {zone}, size {size}: {e}")
                if zone_portfolio_results:
                    zone_portfolio_results.sort(key=lambda x: x[2], reverse=True)
                    best_for_size = zone_portfolio_results[0]
                    zone_overall_results.append((size, best_for_size))
                    if best_for_size[2] > best_zone_fitness:
                        best_zone_fitness = best_for_size[2]
                        best_zone_method = best_for_size[0]
                        best_zone_solution = list(best_for_size[1])
                        best_zone_size = size
                else:
                    logging.warning(f"No valid results for zone {zone} at portfolio size {size}")
            if zone_overall_results:
                zone_overall_results.sort(key=lambda x: x[1][2], reverse=True)
                best_for_zone = zone_overall_results[0]
                best_zone_size = best_for_zone[0]
                best_zone_method = best_for_zone[1][0]
                best_zone_solution = list(best_for_zone[1][1])
                best_zone_fitness = best_for_zone[1][2]
            else:
                logging.error(f"No valid zone-specific portfolio found for zone {zone}")
                if daily_matrix.shape[1] > 0:
                    default_size = min(z_min_size, daily_matrix.shape[1])
                    logging.warning(f"Creating default portfolio of size {default_size} for zone {zone}")
                    strategy_performance = []
                    for i in range(daily_matrix.shape[1]):
                        returns = daily_matrix[:, i]
                        roi = np.sum(returns)
                        strategy_performance.append((i, roi))
                    strategy_performance.sort(key=lambda x: x[1], reverse=True)
                    best_zone_solution = [s[0] for s in strategy_performance[:default_size]]
                    best_zone_method = "Default"
                    best_zone_size = default_size
                    best_zone_fitness = 0.0
                else:
                    continue
            best_zone_portfolio_returns = daily_matrix[:, best_zone_solution].mean(axis=1)
            equity_curve = np.cumsum(best_zone_portfolio_returns)
            total_roi = np.sum(best_zone_portfolio_returns)
            peak = np.maximum.accumulate(equity_curve)
            max_drawdown = np.max(peak - equity_curve) if len(equity_curve) > 0 else 0
            win_days = np.sum(best_zone_portfolio_returns > 0)
            total_days = len(best_zone_portfolio_returns)
            win_percentage = win_days / total_days if total_days > 0 else 0
            pos_sum = np.sum(best_zone_portfolio_returns[best_zone_portfolio_returns > 0]) if any(best_zone_portfolio_returns > 0) else 0
            neg_sum = abs(np.sum(best_zone_portfolio_returns[best_zone_portfolio_returns < 0])) if any(best_zone_portfolio_returns < 0) else 0
            profit_factor = pos_sum / neg_sum if neg_sum != 0 else np.inf
            timestamp = datetime.now().strftime("%d%b%Y%H%M%S")
            zone_results_file = os.path.join(run_dir, f"best_portfolio_zone_{zone}_{timestamp}.txt")
            best_zone_strat_names = [zone_strat_names[i] for i in best_zone_solution]
            with open(zone_results_file, 'w') as f:
                f.write(f"Best Portfolio Summary for Zone {zone}:\n")
                f.write(f"Size: {best_zone_size}\n")
                f.write(f"Method: {best_zone_method}\n")
                f.write(f"Fitness: {best_zone_fitness:.4f}\n\n")
                f.write("Performance Metrics:\n")
                f.write(f"Net Profit: {total_roi:.2f}\n")
                f.write(f"ROI: {total_roi:.2f}%\n")
                f.write(f"Max Drawdown: {max_drawdown:.2f}\n")
                f.write(f"Win Percentage: {win_percentage*100:.2f}%\n")
                f.write(f"Profit Factor: {profit_factor:.2f}\n\n")
                f.write("\nInversion Report: See inversion_report.md in the run directory for details.\n\n")
                f.write("Selected Strategies:\n")
                for i, strat in enumerate(best_zone_strat_names, 1):
                    f.write(f"{i}. {strat}\n")
            try:
                fig = plt.figure(figsize=(12, 6))
            except Exception as e:
                logging.error(f"Error in plotting or processing: {e}")
                fig = None
            plt.plot(equity_curve)
            plt.title(f"Equity Curve for Zone {zone} - Portfolio Size: {best_zone_size}, Method: {best_zone_method}")
            plt.xlabel("Days")
            plt.ylabel("Cumulative Return")
            plt.grid(True)
            zone_equity_curve_file = os.path.join(run_dir, f"equity_curve_zone_{zone}_{timestamp}.png")
            plt.savefig(zone_equity_curve_file)
            plt.close()
            zone_results[zone] = {
                "best_size": best_zone_size,
                "best_method": best_zone_method,
                "best_fitness": best_zone_fitness,
                "best_solution": best_zone_solution,
                "strategy_names": best_zone_strat_names,
                "metrics": {
                    "net_profit": total_roi,
                    "total_roi": total_roi,
                    "max_drawdown": max_drawdown,
                    "win_percentage": win_percentage,
                    "profit_factor": profit_factor
                },
                "output_files": {
                    "Results": zone_results_file,
                    "Equity Curve": zone_equity_curve_file
                },
                "raw_returns": best_zone_portfolio_returns.tolist()
            }
            logging.info(f"Zone-specific optimization completed for zone: {zone}")
    return zone_results

def generate_zone_correlation_analysis(zone_specific_results, run_dir):
    """
    Generate correlation analysis between zone portfolios with proper error handling.
    """
    try:
        zone_returns = {}
        for zone, data in zone_specific_results.items():
            if "raw_returns" in data and data["raw_returns"]:
                zone_returns[zone] = np.array(data["raw_returns"])
        if len(zone_returns) < 2:
            logging.warning(f"Not enough zones with valid returns for correlation analysis: {len(zone_returns)} zones")
            return None
        return_lengths = {zone: len(returns) for zone, returns in zone_returns.items()}
        logging.info(f"Zone return lengths: {return_lengths}")
        if len(set(return_lengths.values())) > 1:
            min_length = min(return_lengths.values())
            logging.warning(f"Zone returns have inconsistent lengths, truncating to {min_length}")
            zone_names = []
            truncated_returns = []
            for zone, returns in zone_returns.items():
                zone_names.append(zone)
                truncated_returns.append(returns[:min_length])
            zone_matrix = np.array(truncated_returns)
        else:
            zone_names = list(zone_returns.keys())
            zone_matrix = np.array(list(zone_returns.values()))
        zone_correlation = np.corrcoef(zone_matrix)
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(zone_correlation, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.title('Correlation Between Zone Portfolios')
        plt.xticks(range(len(zone_names)), zone_names, rotation=45)
        plt.yticks(range(len(zone_names)), zone_names)
        for i in range(len(zone_names)):
            for j in range(len(zone_names)):
                plt.text(j, i, f'{zone_correlation[i, j]:.2f}', ha='center', va='center',
                         color='white' if abs(zone_correlation[i, j]) > 0.5 else 'black')
        plt.tight_layout()
        heatmap_file = os.path.join(run_dir, "zone_correlation_heatmap.png")
        plt.savefig(heatmap_file)
        plt.close()
        return {
            "matrix": zone_correlation.tolist(),
            "zones": zone_names,
            "heatmap_file": heatmap_file
        }
    except Exception as e:
        logging.error(f"Error generating zone correlation analysis: {e}")
        return None

def generate_zone_summary_report_from_zone_reports(zone_reports: Dict[str, str], run_dir: str) -> str:
    """
    Generate a summary report from individual zone reports.
    This version reads the text report files.
    """
    desired_zones = {"Zone 1", "Zone 2", "Zone 3", "Zone 4"}
    report_lines = [
        "# Zone Optimization Summary Report",
        "",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        "This report summarizes the optimization results for each trading zone.",
        "The optimization process found the best combination of strategies for each zone,",
        "maximizing performance according to the selected metric.",
        "",
        "## Zone-Specific Optimization Results"
    ]
    zone_metrics = {}
    for zone, file_path in zone_reports.items():
        if zone in desired_zones:
            report_lines.append(f"### {zone}")
            report_lines.append(f"Report file: {os.path.basename(file_path)}")
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    import re
                    metrics = {}
                    size_match = re.search(r"Size:\s*(\d+)", content)
                    if size_match:
                        metrics["portfolio_size"] = int(size_match.group(1))
                    method_match = re.search(r"Method:\s*(\w+)", content)
                    if method_match:
                        metrics["method"] = method_match.group(1)
                    roi_match = re.search(r"Net Profit:\s*([-+]?\d*\.\d+|\d+)", content)
                    if roi_match:
                        metrics["roi"] = float(roi_match.group(1))
                    dd_match = re.search(r"Max Drawdown:\s*([-+]?\d*\.\d+|\d+)", content)
                    if dd_match:
                        metrics["max_drawdown"] = float(dd_match.group(1))
                    win_match = re.search(r"Win Percentage:\s*([-+]?\d*\.\d+|\d+)", content)
                    if win_match:
                        metrics["win_percentage"] = float(win_match.group(1))
                    pf_match = re.search(r"Profit Factor:\s*([-+]?\d*\.\d+|\d+)", content)
                    if pf_match:
                        metrics["profit_factor"] = float(pf_match.group(1))
                    zone_metrics[zone] = metrics
                    for key, value in metrics.items():
                        formatted_key = key.replace("_", " ").title()
                        if isinstance(value, float):
                            report_lines.append(f"- **{formatted_key}**: {value:.2f}")
                        else:
                            report_lines.append(f"- **{formatted_key}**: {value}")
                    equity_curve_file = os.path.join(os.path.dirname(file_path), f"equity_curve_{zone}_{datetime.now().strftime('%d%b%Y')}.png")
                    if os.path.exists(equity_curve_file):
                        report_lines.append(f"- [Equity Curve]({os.path.basename(equity_curve_file)})")
                    report_lines.append("")
            except Exception as e:
                logging.error(f"Error extracting metrics from {zone} report: {e}")
                report_lines.append(f"- Error extracting metrics: {str(e)}")
                report_lines.append("")
    if len(zone_metrics) > 1:
        report_lines.append("## Zone Comparison")
        report_lines.append("")
        report_lines.append("| Zone | Portfolio Size | Method | ROI | Max Drawdown | Win % | Profit Factor |")
        report_lines.append("|------|---------------|--------|-----|--------------|-------|--------------|")
        for zone in sorted(zone_metrics.keys()):
            metrics = zone_metrics[zone]
            size = metrics.get("portfolio_size", "-")
            method = metrics.get("method", "-")
            roi = f"{metrics.get('roi', 0):.2f}" if "roi" in metrics else "-"
            dd = f"{metrics.get('max_drawdown', 0):.2f}" if "max_drawdown" in metrics else "-"
            win = f"{metrics.get('win_percentage', 0):.2f}" if "win_percentage" in metrics else "-"
            pf = f"{metrics.get('profit_factor', 0):.2f}" if "profit_factor" in metrics else "-"
            report_lines.append(f"| {zone} | {size} | {method} | {roi} | {dd} | {win} | {pf} |")
        report_lines.append("")
    report_lines.append("## Recommendations")
    report_lines.append("")
    best_zone = None
    best_roi = float('-inf')
    for zone, metrics in zone_metrics.items():
        if "roi" in metrics and metrics["roi"] > best_roi:
            best_roi = metrics["roi"]
            best_zone = zone
    if best_zone:
        report_lines.append(f"Based on the optimization results, {best_zone} shows the best performance in terms of ROI.")
        report_lines.append("Consider allocating more capital to this zone's strategy portfolio.")
        report_lines.append("")
    report_lines.append("### Suggested Allocation")
    report_lines.append("")
    report_lines.append("The following allocation is recommended based on relative performance:")
    report_lines.append("")
    total_positive_roi = sum(max(0.01, metrics.get("roi", 0)) for metrics in zone_metrics.values())
    if total_positive_roi > 0:
        report_lines.append("| Zone | Allocation % |")
        report_lines.append("|------|-------------|")
        for zone, metrics in sorted(zone_metrics.items()):
            roi = max(0.01, metrics.get("roi", 0))
            allocation = (roi / total_positive_roi) * 100
            report_lines.append(f"| {zone} | {allocation:.2f}% |")
    else:
        report_lines.append("Unable to calculate meaningful allocation due to non-positive ROI values.")
    summary_report = "\n".join(report_lines)
    summary_file = os.path.join(run_dir, "zone_summary_report.md")
    with open(summary_file, "w") as f:
        f.write(summary_report)
    logging.info(f"Zone summary report generated: {summary_file}")
    return summary_file

# =============================================================================
# Portfolio Optimization Main Function
# =============================================================================
def run_portfolio_optimization(config: Dict[str, Any], output_dir: str,
                               progress_callback: Optional[Callable[[float, str], bool]] = None) -> Optional[Dict[str, Any]]:
    try:
        for section in ['GENERAL', 'PORTFOLIO', 'OPTIMIZATION']:
            if section not in config:
                logging.error(f"Missing '{section}' section in configuration file")
                return {"status": "error", "message": f"Missing '{section}' section in configuration file"}
        if 'PATHS' not in config and ('OPTIMIZATION' in config and str(config['OPTIMIZATION'].get('zone_optimization_enabled', 'False')).lower() != 'true'):
            logging.error("Missing 'PATHS' section in configuration file required for traditional optimization")
            return {"status": "error", "message": "Missing 'PATHS' section in configuration file"}
        if 'ALGORITHMS' not in config:
            logging.warning("Missing 'ALGORITHMS' section, defaulting to Genetic Algorithm only")
            config['ALGORITHMS'] = {'use_genetic_algorithm': 'True'}
        global FILE_LABELS, BALANCED_MODE, DESIRED_RATIO, PENALTY_FACTOR
        global USE_CHECKPOINT, CHECKPOINT_PER_SIZE, PRESERVE_CHECKPOINTS
        USE_CHECKPOINT = safe_get_config_value(config, 'GENERAL', 'use_checkpoint', False, 
                                              lambda x: str(x).lower() in ["true", "1", "yes"])
        CHECKPOINT_PER_SIZE = safe_get_config_value(config, 'GENERAL', 'checkpoint_per_size', True, 
                                                   lambda x: str(x).lower() in ["true", "1", "yes"])
        PRESERVE_CHECKPOINTS = safe_get_config_value(config, 'GENERAL', 'preserve_checkpoints', True, 
                                                    lambda x: str(x).lower() in ["true", "1", "yes"])
        logging.info(f"Checkpoint configuration: Enabled={USE_CHECKPOINT}, Per-Size={CHECKPOINT_PER_SIZE}, Preserve={PRESERVE_CHECKPOINTS}")
        if not PRESERVE_CHECKPOINTS and not USE_CHECKPOINT:
            clean_checkpoints(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        error_log = os.path.join(run_dir, "error_log.txt")
        try:
            error_handler = logging.FileHandler(error_log)
            error_handler.setLevel(logging.ERROR)
            error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            error_handler.setFormatter(error_formatter)
            logging.getLogger().addHandler(error_handler)
        except Exception as e:
            logging.warning(f"Could not set up error log file: {e}")
        strat_limit = config['GENERAL'].get('strat_limit', 'all')
        ga_generations = max(10, int(config['GENERAL'].get('ga_generations', 50)))
        metric = config['GENERAL'].get('metric', 'ratio').lower()
        valid_metrics = ['roi', 'less max dd', 'ratio', 'win percentage', 'profit factor', 'expectancy']
        if metric not in valid_metrics:
            logging.warning(f"Invalid metric '{metric}', defaulting to 'ratio'")
            metric = 'ratio'
        try:
            min_portfolio_size = int(config['PORTFOLIO'].get('min_size', 3))
        except:
            min_portfolio_size = 3
        try:
            max_portfolio_size = int(config['PORTFOLIO'].get('max_size', 5))
        except:
            max_portfolio_size = max(5, min_portfolio_size)
        logging.info(f"Using portfolio size range: min={min_portfolio_size}, max={max_portfolio_size}")
        try:
            population_size = max(10, int(config['PORTFOLIO'].get('population_size', 30)))
        except:
            population_size = 30
        try:
            mutation_rate = max(0.01, min(0.5, float(config['PORTFOLIO'].get('mutation_rate', 0.1))))
        except:
            mutation_rate = 0.1
        BALANCED_MODE = str(config['OPTIMIZATION'].get('balanced_mode', "False")).lower() in ["true", "1", "yes"]
        try:
            PENALTY_FACTOR = max(0, float(config['OPTIMIZATION'].get('penalty_factor', 1.0)))
        except:
            PENALTY_FACTOR = 1.0
        drawdown_threshold = float(config['OPTIMIZATION'].get('drawdown_threshold', 0))
        apply_ulta = str(config['OPTIMIZATION'].get('apply_ulta_logic', "True")).lower() in ["true", "1", "yes"]
        zone_opt_enabled = str(config['OPTIMIZATION'].get('zone_optimization_enabled', "True")).lower() in ["true", "1", "yes"]
        use_ga = str(config['ALGORITHMS'].get('use_genetic_algorithm', "True")).lower() in ["true", "1", "yes"]
        use_pso = str(config['ALGORITHMS'].get('use_particle_swarm', "True")).lower() in ["true", "1", "yes"]
        use_sa = str(config['ALGORITHMS'].get('use_simulated_annealing', "True")).lower() in ["true", "1", "yes"]
        use_de = str(config['ALGORITHMS'].get('use_differential_evolution', "True")).lower() in ["true", "1", "yes"]
        use_aco = str(config['ALGORITHMS'].get('use_ant_colony', "True")).lower() in ["true", "1", "yes"]
        use_hc = str(config['ALGORITHMS'].get('use_hill_climbing', "True")).lower() in ["true", "1", "yes"]
        use_bo = str(config['ALGORITHMS'].get('use_bayesian_optimization', "True")).lower() in ["true", "1", "yes"]
        if not any([use_ga, use_pso, use_sa, use_de, use_aco, use_hc, use_bo]):
            logging.warning("No optimization algorithms selected. Defaulting to GA.")
            use_ga = True
        if BALANCED_MODE:
            for file_ratio in config['FILE_RATIOS'].items():
                DESIRED_RATIO[file_ratio[0]] = int(file_ratio[1])
            logging.info(f"Balanced mode enabled with desired ratios: {DESIRED_RATIO}")
        if zone_opt_enabled:
            if "consolidated_dir" not in config['PATHS']:
                logging.error("Zone optimization enabled but no consolidated_dir specified in config")
                return None
            consolidated_dir = config['PATHS']['consolidated_dir']
            if not os.path.exists(consolidated_dir):
                logging.error(f"Consolidated directory not found: {consolidated_dir}")
                return None
            logging.info(f"Zone optimization enabled. Using consolidated data from {consolidated_dir}")
            try:
                consolidated_df = load_consolidated_df_from_directory(consolidated_dir)
                unique_zones = consolidated_df["Zone"].unique()
                logging.info(f"Found {len(unique_zones)} unique zones: {unique_zones}")
                zone_matrix, strategy_cols = build_zone_matrix_from_df(consolidated_df)
                calibration_params = calibrate_zone_fitness_parameters(zone_matrix)
                global MAX_CONSISTENCY, MEAN_CONSISTENCY, MAX_CORRELATION, MEAN_CORRELATION
                MAX_CONSISTENCY = calibration_params["max_consistency"]
                MEAN_CONSISTENCY = calibration_params["mean_consistency"]
                MAX_CORRELATION = calibration_params["max_correlation"]
                MEAN_CORRELATION = calibration_params["mean_correlation"]
                zone_weights = None
                if "ZONE_WEIGHTS" in config:
                    try:
                        zone_weights_dict = {zone.lower().replace(" ", ""): float(weight) 
                                             for zone, weight in config["ZONE_WEIGHTS"].items()}
                        all_zones = sorted(set(zone.lower().replace(" ", "") for zone in unique_zones))
                        zone_weights = np.array([zone_weights_dict.get(zone, 1.0) for zone in all_zones])
                        zone_weights = zone_weights / np.sum(zone_weights)
                        logging.info(f"Using zone weights: {dict(zip(all_zones, zone_weights))}")
                    except Exception as e:
                        logging.error(f"Error parsing zone weights: {e}. Using uniform weights.")
                        zone_weights = np.ones(len(all_zones)) / len(all_zones)
                if "ZONE_SPECIFIC_OPTIMIZATION" in config and str(config["ZONE_SPECIFIC_OPTIMIZATION"].get("enable", "True")).lower() in ["true", "1", "yes"]:
                    logging.info("Running zone-specific optimization for each zone")
                    try:
                        zone_specific_results = run_zone_specific_optimization_for_all_zones(
                            consolidated_df, run_dir, config, progress_callback)
                        corr_analysis = generate_zone_correlation_analysis(zone_specific_results, run_dir)
                        if corr_analysis:
                            logging.info(f"Zone correlation analysis saved to {corr_analysis['heatmap_file']}")
                        zone_reports = {zone: data["output_files"]["Results"] 
                                        for zone, data in zone_specific_results.items()
                                        if "output_files" in data and "Results" in data["output_files"]}
                        if zone_reports:
                            summary_file = generate_zone_summary_report_from_zone_reports(zone_reports, run_dir)
                            logging.info(f"Zone summary report generated: {summary_file}")
                    except Exception as e:
                        logging.error(f"Error in zone-specific optimization: {e}")
                        logging.error(traceback.format_exc())
                if use_ga:
                    for size in range(min_portfolio_size, max_portfolio_size + 1):
                        logging.info(f"Running GA-Zone optimization for portfolio size {size}")
                        if progress_callback:
                            progress_callback(0, f"Running GA-Zone optimization for portfolio size {size}")
                        try:
                            best_solution, best_fitness = genetic_algorithm_zone(
                                zone_matrix, size, metric, np.eye(zone_matrix.shape[2]), run_dir,
                                zone_weights, generations=ga_generations, population_size=population_size,
                                mutation_rate=mutation_rate, progress_callback=progress_callback)
                            if best_solution:
                                best_strategy_names = [strategy_cols[i] for i in best_solution]
                                logging.info(f"Best zone portfolio (size {size}): {best_strategy_names}")
                                logging.info(f"Best zone fitness: {best_fitness}")
                                all_zone_returns = []
                                for z in range(zone_matrix.shape[1]):
                                    zone_name = unique_zones[z] if z < len(unique_zones) else f"Zone {z+1}"
                                    zone_returns = np.mean(zone_matrix[:, z, best_solution], axis=1)
                                    all_zone_returns.append(zone_returns)
                                    zone_equity_curve = np.cumsum(zone_returns)
                                    try:
                                        fig = plt.figure(figsize=(10, 6))
                                    except Exception as e:
                                        logging.error(f"Error in plotting or processing: {e}")
                                        fig = None
                                    plt.plot(zone_equity_curve)
                                    plt.title(f"Equity Curve for {zone_name} - Portfolio Size: {size}")
                                    plt.xlabel("Days")
                                    plt.ylabel("Cumulative Return")
                                    plt.grid(True)
                                    zone_equity_file = os.path.join(
                                        run_dir, f"ga_zone_equity_{zone_name}_{size}_{run_id}.png")
                                    plt.savefig(zone_equity_file)
                                    plt.close()
                                if zone_weights is not None and len(zone_weights) == len(all_zone_returns):
                                    weighted_returns = np.zeros(len(all_zone_returns[0]))
                                    for z, returns in enumerate(all_zone_returns):
                                        weighted_returns += returns * zone_weights[z]
                                else:
                                    weighted_returns = np.mean(all_zone_returns, axis=0)
                                portfolio_report = generate_performance_report(
                                    weighted_returns, f"GA_Zone_Portfolio_Size{size}", run_dir)
                                with open(os.path.join(run_dir, f"ga_zone_portfolio_size{size}_{run_id}.txt"), 'w') as f:
                                    f.write(f"GA Zone Portfolio (Size {size})\n")
                                    f.write(f"Fitness: {best_fitness}\n\n")
                                    f.write("Selected Strategies:\n")
                                    for i, strat in enumerate(best_strategy_names, 1):
                                        f.write(f"{i}. {strat}\n")
                                    f.write("\nPerformance Metrics:\n")
                                    for key, value in portfolio_report["metrics"].items():
                                        f.write(f"{key}: {value}\n")
                        except Exception as e:
                            logging.error(f"Error in GA-Zone for size {size}: {e}")
                if use_pso:
                    for size in range(min_portfolio_size, max_portfolio_size + 1):
                        logging.info(f"Running PSO-Zone optimization for portfolio size {size}")
                        if progress_callback:
                            progress_callback(0, f"Running PSO-Zone optimization for portfolio size {size}")
                        try:
                            best_solution, best_fitness = pso_algorithm_zone(
                                zone_matrix, size, metric, np.eye(zone_matrix.shape[2]), run_dir,
                                iterations=ga_generations, swarm_size=population_size, zone_weights=zone_weights,
                                progress_callback=progress_callback)
                            if best_solution:
                                best_strategy_names = [strategy_cols[i] for i in best_solution]
                                logging.info(f"Best PSO zone portfolio (size {size}): {best_strategy_names}")
                                logging.info(f"Best PSO zone fitness: {best_fitness}")
                                with open(os.path.join(run_dir, f"pso_zone_portfolio_size{size}_{run_id}.txt"), 'w') as f:
                                    f.write(f"PSO Zone Portfolio (Size {size})\n")
                                    f.write(f"Fitness: {best_fitness}\n\n")
                                    f.write("Selected Strategies:\n")
                                    for i, strat in enumerate(best_strategy_names, 1):
                                        f.write(f"{i}. {strat}\n")
                        except Exception as e:
                            logging.error(f"Error in PSO-Zone for size {size}: {e}")
                return {"status": "success", "output_dir": run_dir, "run_id": run_id}
            except Exception as e:
                logging.error(f"Error in zone optimization: {e}")
                logging.error(traceback.format_exc())
                return None
        logging.info("Entire-day mode optimization")
        data_folder = config['PATHS']['data_folder']
        if not os.path.exists(data_folder):
            logging.error(f"Data folder not found: {data_folder}")
            return None
        try:
            daily_df, file_labels = load_and_merge_files(data_folder, strat_limit)
            FILE_LABELS = file_labels
            if apply_ulta:
                logging.info("Applying ULTA logic to invert negative strategies")
                daily_df, inverted_strategies = apply_ulta_logic(daily_df)
                if inverted_strategies:
                    inversion_report = generate_inversion_report(inverted_strategies)
                    inversion_report_file = os.path.join(run_dir, "inversion_report.md")
                    with open(inversion_report_file, 'w') as f:
                        f.write(inversion_report)
                    logging.info(f"Inversion report written to {inversion_report_file}")
            strategy_cols = [col for col in daily_df.columns if col not in ["Date", "Zone", "Day"]]
            try:
                daily_matrix = daily_df[strategy_cols].to_numpy().astype(float)
            except Exception as e:
                logging.error(f"Error converting to numeric array: {e}")
                num_data = []
                for col in strategy_cols:
                    try:
                        num_data.append(pd.to_numeric(daily_df[col], errors='coerce').fillna(0).values)
                    except Exception as e2:
                        num_data.append(np.zeros(len(daily_df)))
                daily_matrix = np.column_stack(num_data)
            try:
                corr_matrix = np.corrcoef(daily_matrix.T)
                corr_matrix = np.nan_to_num(corr_matrix)
            except Exception as e:
                logging.error(f"Error calculating correlation matrix: {e}")
                corr_matrix = np.eye(daily_matrix.shape[1])
            logging.info("Evaluating individual strategy performance...")
            strat_metrics = evaluate_individual_strategies_parallel(daily_df)
            strat_metrics_file = os.path.join(run_dir, "strategy_metrics.csv")
            strat_metrics.to_csv(strat_metrics_file)
            logging.info(f"Strategy metrics saved to {strat_metrics_file}")
            from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
            # Use maximum CPU utilization for parallel processing
            max_workers = min(os.cpu_count() or 1, 8)  # Use all available cores, max 8
            logging.info(f"Using parallel processing with {max_workers} workers for maximum CPU utilization")
            
            # Use ThreadPoolExecutor on Windows for better compatibility
            if os.name == 'nt':
                logging.info("Windows detected - using ThreadPoolExecutor for better compatibility")
                executor = ThreadPoolExecutor(max_workers=max_workers)
            else:
                # Use ProcessPoolExecutor on non-Windows systems
                executor = ProcessPoolExecutor(max_workers=max_workers)
            
            try:
                best_overall_solution = None
                best_overall_fitness = float('-inf')
                best_overall_size = 0
                best_overall_method = ""
                for size in range(min_portfolio_size, max_portfolio_size + 1):
                    logging.info(f"Optimizing for portfolio size {size}")
                    if progress_callback:
                        progress_callback(0, f"Optimizing for portfolio size {size}")
                    if size > daily_matrix.shape[1]:
                        logging.warning(f"Portfolio size {size} exceeds available strategies ({daily_matrix.shape[1]})")
                        continue
                    portfolio_results = []
                    if use_ga:
                        try:
                            logging.info(f"Running genetic algorithm for size {size}...")
                            sol, fit = genetic_algorithm(daily_matrix, size, metric, corr_matrix, run_dir,
                                                         generations=ga_generations, population_size=population_size,
                                                         mutation_rate=mutation_rate, drawdown_threshold=drawdown_threshold,
                                                         executor=executor, progress_callback=progress_callback)
                            portfolio_results.append(("GA", sol, fit))
                        except Exception as e:
                            logging.error(f"Error in GA for size {size}: {e}")
                    if use_pso:
                        try:
                            logging.info(f"Running particle swarm optimization for size {size}...")
                            sol, fit = pso_algorithm(daily_matrix, size, metric, corr_matrix, run_dir,
                                                     iterations=ga_generations, swarm_size=population_size,
                                                     drawdown_threshold=drawdown_threshold, executor=executor,
                                                     progress_callback=progress_callback)
                            portfolio_results.append(("PSO", sol, fit))
                        except Exception as e:
                            logging.error(f"Error in PSO for size {size}: {e}")
                    if use_sa:
                        try:
                            logging.info(f"Running simulated annealing for size {size}...")
                            sol, fit = simulated_annealing(daily_matrix, size, metric, corr_matrix, run_dir,
                                                           iterations=1000, drawdown_threshold=drawdown_threshold,
                                                           progress_callback=progress_callback)
                            portfolio_results.append(("SA", sol, fit))
                        except Exception as e:
                            logging.error(f"Error in SA for size {size}: {e}")
                    if use_de:
                        try:
                            logging.info(f"Running differential evolution for size {size}...")
                            sol, fit = differential_evolution(daily_matrix, size, metric, corr_matrix, run_dir,
                                                             population_size=population_size, iterations=ga_generations,
                                                             drawdown_threshold=drawdown_threshold,
                                                             progress_callback=progress_callback)
                            portfolio_results.append(("DE", sol, fit))
                        except Exception as e:
                            logging.error(f"Error in DE for size {size}: {e}")
                    if use_aco:
                        try:
                            logging.info(f"Running ant colony optimization for size {size}...")
                            sol, fit = ant_colony_optimization(daily_matrix, size, metric, corr_matrix, run_dir,
                                                              iterations=ga_generations, drawdown_threshold=drawdown_threshold,
                                                              progress_callback=progress_callback)
                            portfolio_results.append(("ACO", sol, fit))
                        except Exception as e:
                            logging.error(f"Error in ACO for size {size}: {e}")
                    if use_hc:
                        try:
                            logging.info(f"Running hill climbing for size {size}...")
                            sol, fit = hill_climbing(daily_matrix, size, metric, corr_matrix, run_dir,
                                                      iterations=200, drawdown_threshold=drawdown_threshold,
                                                      progress_callback=progress_callback)
                            portfolio_results.append(("HC", sol, fit))
                        except Exception as e:
                            logging.error(f"Error in HC for size {size}: {e}")
                    if use_bo:
                        try:
                            logging.info(f"Running Bayesian optimization for size {size}...")
                            sol, fit = bayesian_optimization(daily_matrix, size, metric, corr_matrix, run_dir,
                                                            iterations=ga_generations, drawdown_threshold=drawdown_threshold,
                                                            progress_callback=progress_callback)
                            portfolio_results.append(("BO", sol, fit))
                        except Exception as e:
                            logging.error(f"Error in BO for size {size}: {e}")
                    if portfolio_results:
                        portfolio_results.sort(key=lambda x: x[2], reverse=True)
                        best_method, best_solution, best_fitness = portfolio_results[0]
                        best_portfolio_returns = daily_matrix[:, best_solution].mean(axis=1)
                        perf_report = generate_performance_report(
                            best_portfolio_returns, f"Best_Portfolio_Size{size}", run_dir)
                        best_strategy_names = [strategy_cols[i] for i in best_solution]
                        portfolio_file = os.path.join(run_dir, f"best_portfolio_size{size}_{run_id}.txt")
                        with open(portfolio_file, 'w') as f:
                            f.write(f"Best Portfolio (Size {size}) - Method: {best_method}\n")
                            f.write(f"Fitness: {best_fitness}\n\n")
                            f.write("Performance Metrics:\n")
                            for key, value in perf_report["metrics"].items():
                                f.write(f"{key}: {value}\n")
                            f.write("\nSelected Strategies:\n")
                            for i, strat in enumerate(best_strategy_names, 1):
                                f.write(f"{i}. {strat}\n")
                        if best_fitness > best_overall_fitness:
                            best_overall_fitness = best_fitness
                            best_overall_solution = best_solution
                            best_overall_size = size
                            best_overall_method = best_method
                if best_overall_solution is not None:
                    logging.info(f"Best overall portfolio size: {best_overall_size}, Method: {best_overall_method}")
                    logging.info(f"Best overall fitness: {best_overall_fitness}")
                    best_strategy_names = [strategy_cols[i] for i in best_overall_solution]
                    best_portfolio_returns = daily_matrix[:, best_overall_solution].mean(axis=1)
                    summary_file = os.path.join(run_dir, f"optimization_summary_{run_id}.txt")
                    with open(summary_file, 'w') as f:
                        f.write("Multi-Strategy Portfolio Optimization Results\n")
                        f.write("===========================================\n\n")
                        f.write(f"Run ID: {run_id}\n")
                        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        f.write("Optimization Parameters:\n")
                        f.write(f"- Metric: {metric}\n")
                        f.write(f"- Min Portfolio Size: {min_portfolio_size}\n")
                        f.write(f"- Max Portfolio Size: {max_portfolio_size}\n")
                        f.write(f"- Population Size: {population_size}\n")
                        f.write(f"- Mutation Rate: {mutation_rate}\n")
                        f.write(f"- GA Generations: {ga_generations}\n")
                        f.write(f"- Apply ULTA Logic: {apply_ulta}\n")
                        f.write(f"- Balanced Mode: {BALANCED_MODE}\n")
                        f.write(f"- Penalty Factor: {PENALTY_FACTOR}\n\n")
                        f.write("Best Overall Portfolio:\n")
                        f.write(f"- Size: {best_overall_size}\n")
                        f.write(f"- Method: {best_overall_method}\n")
                        f.write(f"- Fitness: {best_overall_fitness}\n\n")
                        f.write("Selected Strategies:\n")
                        for i, strat in enumerate(best_strategy_names, 1):
                            f.write(f"{i}. {strat}\n")
                    logging.info(f"Optimization summary saved to {summary_file}")
                    eq_curve = np.cumsum(best_portfolio_returns)
                    total_roi = np.sum(best_portfolio_returns)
                    peak = np.maximum.accumulate(eq_curve)
                    max_dd = np.max(peak - eq_curve) if len(eq_curve) > 0 else 0
                    win_pct = np.sum(best_portfolio_returns > 0) / len(best_portfolio_returns) if len(best_portfolio_returns) > 0 else 0
                    return {
                        "status": "success",
                        "output_dir": run_dir,
                        "run_id": run_id,
                        "best_portfolio": {
                            "size": best_overall_size,
                            "method": best_overall_method,
                            "fitness": best_overall_fitness,
                            "strategy_indices": best_overall_solution,
                            "strategy_names": best_strategy_names,
                            "metrics": {
                                "total_roi": total_roi,
                                "max_drawdown": max_dd,
                                "win_percentage": win_pct
                            }
                        }
                    }
                else:
                    logging.error("No valid portfolio found")
                    return {"status": "error", "message": "No valid portfolio found"}
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)
        except Exception as e:
            logging.error(f"Error in portfolio optimization: {e}")
            logging.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}
    except Exception as e:
        logging.error(f"Error in run_portfolio_optimization: {e}")
        logging.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

# If run as a standalone script
if __name__ == "__main__":
    import argparse
    import configparser
    import os
    import matplotlib
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description="Multi-Strategy Portfolio Optimizer")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--output", "-o", type=str, default="./output", help="Output directory")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found.")
        dir_path = os.path.dirname(args.config) if os.path.dirname(args.config) else '.'
        filename = os.path.basename(args.config)
        similar_files = [f for f in os.listdir(dir_path) if f.endswith('.ini')]
        if similar_files:
            print("Did you mean one of these?")
            for f in similar_files:
                print(f"  - {f}")
        sys.exit(1)
    config = configparser.ConfigParser()
    try:
        config.read(args.config)
        required_sections = ['GENERAL', 'PORTFOLIO', 'OPTIMIZATION']
        missing_sections = [section for section in required_sections if section not in config]
        if missing_sections:
            print(f"Error: Missing required sections in config file: {', '.join(missing_sections)}")
            sys.exit(1)
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        sys.exit(1)
    gpu_available = setup_environment()
    try:
        result = run_portfolio_optimization(config, args.output)
        if result and result["status"] == "success":
            print(f"Optimization completed successfully. Results saved to {result['output_dir']}")
        else:
            error_msg = result.get("message", "Unknown error") if result else "Unknown error"
            print(f"Optimization failed: {error_msg}")
    except Exception as e:
        import traceback
        print(f"Unhandled exception during optimization: {e}")
        traceback.print_exc()
    finally:
        plt.close('all')

import gc
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from prediction_common import StockCalendar, HTMLUpdater, GitOperator, load_model
import quantlab.data.tencent_5min_download as mins_download

# --- Configuration ---
Config = {
    "REPO_PATH": Path("./examples/demo"),
    "MODEL_PATH": "./examples/demo/models",
    "SYMBOL": 'SH601600',
    "INTERVAL": '5min',
    "HIST_POINTS": 420,
    "PRED_HORIZON": 50,
    "N_PREDICTIONS": 30,
    "VOL_WINDOW": 50,
}


def make_prediction(df, predictor):
    """Generates probabilistic forecasts using the Kronos model,
    with timestamps aligned to A-share trading hours."""
    last_timestamp = df['timestamps'].max()
    step = pd.Timedelta(minutes=5)
    calendar = StockCalendar()

    new_timestamps = []
    t = last_timestamp
    while len(new_timestamps) < Config["PRED_HORIZON"]:
        t = calendar.next_bar(t, step)
        new_timestamps.append(t)
    new_timestamps_index = pd.DatetimeIndex(new_timestamps)

    y_timestamp = pd.Series(new_timestamps_index, name='y_timestamp')
    x_timestamp = df['timestamps']
    x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]

    with torch.no_grad():
        print("Making main prediction (T=1.0)...")
        begin_time = time.time()
        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=Config["PRED_HORIZON"], T=1.0, top_p=0.95,
            sample_count=Config["N_PREDICTIONS"], verbose=True
        )
        print(f"Main prediction completed in {time.time() - begin_time:.2f} seconds.")
        close_preds_main = pred_df['close']
        if isinstance(close_preds_main, pd.Series):
            close_preds_main = close_preds_main.to_frame()
        volume_preds_main = pred_df['volume']
        if isinstance(volume_preds_main, pd.Series):
            volume_preds_main = volume_preds_main.to_frame()
        close_preds_volatility = close_preds_main

    return close_preds_main, volume_preds_main, close_preds_volatility


def fetch_stock_data():
    """Fetches 5-minute K-line data for A-share stocks from Tencent."""
    symbol = Config["SYMBOL"]
    limit = Config["HIST_POINTS"] + Config["VOL_WINDOW"]

    code = mins_download.qlib_to_tencent(symbol)
    if code is None:
        raise ValueError(f"Invalid stock symbol: {symbol}")

    print(f"Downloading {symbol} 5min data from Tencent ...")
    payload = mins_download.fetch_json(code, 640, timeout=20.0)

    item = payload.get("data", {}).get(code, {})
    bars = item.get("m5") or []
    if not bars:
        raise ValueError(f"No m5 data in response for {symbol}")

    today = datetime.now().strftime("%Y-%m-%d")
    df = mins_download.normalize_rows(symbol, bars, "2000-01-01", today)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    df.rename(columns={'date': 'timestamps'}, inplace=True)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]
    df = df[df['timestamps'].apply(StockCalendar.in_trading_hours)].reset_index(drop=True)

    print(f"Data fetched successfully. {len(df)} bars loaded.")
    return df.tail(limit).reset_index(drop=True)


def calculate_metrics(hist_df, close_preds_df, v_close_preds_df):
    """Calculates upside and volatility amplification probabilities."""
    last_close = hist_df['close'].iloc[-1]

    final_hour_preds = close_preds_df.iloc[-1]
    upside_prob = (final_hour_preds > last_close).mean()

    hist_log_returns = np.log(hist_df['close'] / hist_df['close'].shift(1))
    historical_vol = hist_log_returns.iloc[-Config["VOL_WINDOW"]:].std()

    amplification_count = 0
    for col in v_close_preds_df.columns:
        full_sequence = pd.concat([pd.Series([last_close]), v_close_preds_df[col]]).reset_index(drop=True)
        pred_log_returns = np.log(full_sequence / full_sequence.shift(1))
        predicted_vol = pred_log_returns.std()
        if predicted_vol > historical_vol:
            amplification_count += 1

    vol_amp_prob = amplification_count / len(v_close_preds_df.columns)

    print(f"Upside Probability: {upside_prob:.2%}, Volatility Amplification Probability: {vol_amp_prob:.2%}")
    return upside_prob, vol_amp_prob


def create_plot(hist_df, close_preds_df, volume_preds_df):
    """Generates and saves a comprehensive forecast chart for 5-min A-share data."""
    print("Generating comprehensive forecast chart...")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    hist_time = hist_df['timestamps']
    last_hist_time = hist_time.iloc[-1]
    step = pd.Timedelta(minutes=5)
    calendar = StockCalendar()

    pred_time = []
    t = last_hist_time
    for _ in range(len(close_preds_df)):
        t = calendar.next_bar(t, step)
        pred_time.append(t)

    all_time = list(hist_time) + pred_time
    n_hist = len(hist_time)
    x_hist = np.arange(n_hist)
    x_pred = np.arange(n_hist, n_hist + len(pred_time))

    ax1.plot(x_hist, hist_df['close'].values, color='royalblue', label='Historical Price', linewidth=1.5)
    mean_preds = close_preds_df.mean(axis=1).values
    ax1.plot(x_pred, mean_preds, color='darkorange', linestyle='-', label='Mean Forecast')
    ax1.fill_between(x_pred,
                     close_preds_df.min(axis=1).values, close_preds_df.max(axis=1).values,
                     color='darkorange', alpha=0.2, label='Forecast Range (Min-Max)')
    ax1.set_title(f'{Config["SYMBOL"]} A-Share 5min Probabilistic Price & Volume Forecast', fontsize=16, weight='bold')
    ax1.set_ylabel('Price (CNY)')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax2.bar(x_hist, hist_df['volume'].values, color='skyblue', label='Historical Volume', width=0.8)
    ax2.bar(x_pred, volume_preds_df.mean(axis=1).values, color='sandybrown',
            label='Mean Forecasted Volume', width=0.8)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Time')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    for ax in [ax1, ax2]:
        ax.axvline(x=n_hist - 0.5, color='red', linestyle='--', linewidth=1.5, label='_nolegend_')

    n_ticks = min(20, len(all_time))
    tick_pos = np.linspace(0, len(all_time) - 1, n_ticks, dtype=int)
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels([pd.Timestamp(all_time[i]).strftime('%m/%d\n%H:%M') for i in tick_pos])
    ax2.tick_params(axis='x', rotation=0)

    fig.tight_layout()
    chart_path = Config["REPO_PATH"] / 'prediction_chart.png'
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    print(f"Chart saved to: {chart_path}")





def main_task(model):
    """Executes one full update cycle."""
    print("\n" + "=" * 60 + f"\nStarting update task at {datetime.now(timezone.utc)}\n" + "=" * 60)
    config_obj = SimpleNamespace(REPO_PATH=Config["REPO_PATH"])
    html_updater = HTMLUpdater(config_obj)
    git_operator = GitOperator(config_obj)

    df_full = fetch_stock_data()
    df_for_model = df_full.copy()

    close_preds, volume_preds, v_close_preds = make_prediction(df_for_model, model)

    hist_df_for_plot = df_for_model.tail(Config["HIST_POINTS"])
    hist_df_for_metrics = df_for_model.tail(Config["VOL_WINDOW"])

    upside_prob, vol_amp_prob = calculate_metrics(hist_df_for_metrics, close_preds, v_close_preds)
    create_plot(hist_df_for_plot, close_preds, volume_preds)
    html_updater.update(upside_prob, vol_amp_prob)

    commit_message = f"Auto-update forecast for {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"

    del df_full, df_for_model, close_preds, volume_preds, v_close_preds
    del hist_df_for_plot, hist_df_for_metrics

    gc.collect()

    print("-" * 60 + "\n--- Task completed successfully ---\n" + "-" * 60 + "\n")


def run_scheduler(model):
    """A continuous scheduler that runs the main task with A-share trading session awareness."""
    while True:
        now = datetime.now(timezone.utc) + timedelta(hours=8)
        next_run = now + timedelta(hours=1)
        next_run = next_run.replace(minute=5, second=0, microsecond=0)
        sleep_seconds = (next_run - now).total_seconds()

        if sleep_seconds > 0:
            print(f"Current time (CST): {now:%Y-%m-%d %H:%M:%S}.")
            print(f"Next run at: {next_run:%Y-%m-%d %H:%M:%S}. Waiting for {sleep_seconds:.0f} seconds...")
            time.sleep(sleep_seconds)

        try:
            main_task(model)
        except Exception as e:
            print(f"\n!!!!!! A critical error occurred in the main task !!!!!!!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("Retrying in 5 minutes...")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            time.sleep(300)


if __name__ == '__main__':
    model_path = Path(Config["MODEL_PATH"])
    model_path.mkdir(parents=True, exist_ok=True)

    loaded_model = load_model(Config["MODEL_PATH"], device="cuda:0")
    main_task(loaded_model)

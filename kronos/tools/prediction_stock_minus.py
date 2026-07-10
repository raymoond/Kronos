import gc
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from tools.prediction_common import (
    StockCalendar, HTMLUpdater, GitOperator, load_model,
    make_prediction, calculate_metrics, create_plot,
)
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


def main_task(model):
    """Executes one full update cycle."""
    print("\n" + "=" * 60 + f"\nStarting update task at {datetime.now(timezone.utc)}\n" + "=" * 60)
    config_obj = SimpleNamespace(REPO_PATH=Config["REPO_PATH"])
    html_updater = HTMLUpdater(config_obj)

    df_full = fetch_stock_data()
    df_for_model = df_full.copy()

    close_preds, volume_preds, v_close_preds = make_prediction(
        df_for_model, model,
        pred_horizon=Config["PRED_HORIZON"],
        n_predictions=Config["N_PREDICTIONS"],
    )

    hist_df_for_plot = df_for_model.tail(Config["HIST_POINTS"])
    hist_df_for_metrics = df_for_model.tail(Config["VOL_WINDOW"])

    upside_prob, vol_amp_prob = calculate_metrics(
        hist_df_for_metrics, close_preds, v_close_preds,
        vol_window=Config["VOL_WINDOW"],
    )
    create_plot(hist_df_for_plot, close_preds, volume_preds,
                repo_path=Config["REPO_PATH"], symbol=Config["SYMBOL"])
    html_updater.update(upside_prob, vol_amp_prob)

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

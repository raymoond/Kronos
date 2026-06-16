import gc
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from binance.client import Client
from requests.exceptions import ConnectTimeout, ConnectionError

from model import KronosTokenizer, Kronos, KronosPredictor
import quantlab.data.tencent_5min_download as mins_download


# ============================================================================
# Config
# ============================================================================

@dataclass
class Config:
    REPO_PATH: Path = Path("./examples/demo")
    MODEL_PATH: str = "./examples/demo/models"
    SYMBOL: str = 'BTCUSDT'
    INTERVAL: str = '1h'
    HIST_POINTS: int = 360
    PRED_HORIZON: int = 24
    N_PREDICTIONS: int = 30
    VOL_WINDOW: int = 24

    def interval_min(self) -> int:
        s = self.INTERVAL
        if s.endswith('min'):
            return int(s.replace('min', ''))
        if s.endswith('h'):
            return int(s.replace('h', ''))
        return 60

    def interval_step(self) -> pd.Timedelta:
        s = self.INTERVAL
        if s.endswith('min'):
            return pd.Timedelta(minutes=int(s.replace('min', '')))
        if s.endswith('h'):
            return pd.Timedelta(hours=int(s.replace('h', '')))
        return pd.Timedelta(hours=1)

    def interval_freq(self) -> str:
        s = self.INTERVAL
        if s.endswith('min'):
            return s
        if s.endswith('h'):
            return s
        return 'h'

    def cache_path(self, symbol=None, interval=None) -> Path:
        return self.REPO_PATH / f"{symbol or self.SYMBOL}_{interval or self.INTERVAL}.csv"

    def limit(self) -> int:
        return self.HIST_POINTS + self.VOL_WINDOW


# ============================================================================
# Calendar (abstract product)
# ============================================================================

class Calendar(ABC):
    @abstractmethod
    def next_bar(self, current_ts: pd.Timestamp, step: pd.Timedelta) -> pd.Timestamp:
        ...


class CryptoCalendar(Calendar):
    def next_bar(self, current_ts: pd.Timestamp, step: pd.Timedelta) -> pd.Timestamp:
        return current_ts + step


class StockCalendar(Calendar):
    @staticmethod
    def is_trading_day(t: pd.Timestamp) -> bool:
        return t.weekday() < 5

    @staticmethod
    def in_trading_hours(t: pd.Timestamp) -> bool:
        if not StockCalendar.is_trading_day(t):
            return False
        tm = t.time()
        morning_start = pd.Timestamp('09:30').time()
        morning_end = pd.Timestamp('11:30').time()
        afternoon_start = pd.Timestamp('13:00').time()
        afternoon_end = pd.Timestamp('15:00').time()
        return (morning_start <= tm <= morning_end) or (afternoon_start <= tm <= afternoon_end)

    def next_bar(self, current_ts: pd.Timestamp, step: pd.Timedelta) -> pd.Timestamp:
        t = current_ts + step
        morning_start = pd.Timestamp('09:30').time()
        morning_end = pd.Timestamp('11:30').time()
        afternoon_start = pd.Timestamp('13:00').time()
        afternoon_end = pd.Timestamp('15:00').time()

        while True:
            if t.weekday() >= 5:
                t += pd.Timedelta(days=(7 - t.weekday()))
                t = t.replace(hour=9, minute=30, second=0, microsecond=0)
                continue
            tm = t.time()
            if tm < morning_start:
                t = t.replace(hour=9, minute=30, second=0, microsecond=0)
            elif morning_end < tm < afternoon_start:
                t = t.replace(hour=13, minute=0, second=0, microsecond=0)
            elif tm > afternoon_end:
                t += pd.Timedelta(days=1)
                t = t.replace(hour=9, minute=30, second=0, microsecond=0)
            else:
                return t


# ============================================================================
# DataFetcher (abstract product)
# ============================================================================

class DataFetcher(ABC):
    @abstractmethod
    def fetch(self, config: Config) -> pd.DataFrame:
        ...


class CryptoDataFetcher(DataFetcher):
    def fetch(self, config: Config) -> pd.DataFrame:
        symbol, interval = config.SYMBOL, config.INTERVAL
        limit = config.limit()

        try:
            print(f"Fetching {limit} bars of {symbol} {interval} data from Binance...")
            client = Client()
            client.session.timeout = (10, 30)
            klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

            cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                    'taker_buy_quote_asset_volume', 'ignore']
            df = pd.DataFrame(klines, columns=cols)
            df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']]
            df.rename(columns={'quote_asset_volume': 'amount', 'open_time': 'timestamps'}, inplace=True)
            df['timestamps'] = pd.to_datetime(df['timestamps'], unit='ms', utc=True)
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                df[col] = pd.to_numeric(df[col])

            cache_path = config.cache_path()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)

            data_end = df['timestamps'].max()
            print(f"Data fetched successfully. Latest bar: {data_end}")
            return df

        except (ConnectTimeout, ConnectionError) as e:
            print(f"\n !!! Network error connecting to Binance: {e}")
            print(f" !!! Falling back to local cached data.\n")
            return self._load_cache(config, symbol, interval, limit)

    def _load_cache(self, config, symbol, interval, limit):
        path = config.cache_path(symbol, interval)
        if not path.exists():
            raise FileNotFoundError(f"Local cache file not found: {path}")

        print(f"Loading {limit} bars from local cache: {path}")
        df = pd.read_csv(path)
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[col] = pd.to_numeric(df[col])

        df = df.tail(limit).reset_index(drop=True)
        data_start = df['timestamps'].min()
        data_end = df['timestamps'].max()
        hours_behind = (datetime.now(timezone.utc) - data_end).total_seconds() / 3600
        print(f" !!! WARNING: Using cached data from {data_start} to {data_end}")
        print(f" !!! Data is approximately {hours_behind:.1f} hours behind current time.")
        print(f" !!! Binance API was unreachable. Predictions may be less accurate.\n")
        return df


class StockDataFetcher(DataFetcher):
    def fetch(self, config: Config) -> pd.DataFrame:
        symbol = config.SYMBOL
        limit = config.limit()

        try:
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

            cache_path = config.cache_path(symbol, '5min')
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)

            data_end = df['timestamps'].max()
            print(f"Data fetched successfully. Latest bar: {data_end}")
            return df.tail(limit).reset_index(drop=True)

        except Exception as e:
            print(f"\n !!! Error fetching from Tencent: {e}")
            print(f" !!! Falling back to local cached data.\n")
            return self._load_cache(config, symbol, '5min', limit)

    def _load_cache(self, config, symbol, interval, limit):
        path = config.cache_path(symbol, interval)
        if not path.exists():
            raise FileNotFoundError(f"Local cache file not found: {path}")

        print(f"Loading {limit} bars from local cache: {path}")
        df = pd.read_csv(path)
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[col] = pd.to_numeric(df[col])

        df = df.tail(limit).reset_index(drop=True)
        data_start = df['timestamps'].min()
        data_end = df['timestamps'].max()
        if data_end.tz is None:
            data_end = data_end.tz_localize('Asia/Shanghai')
        mins_behind = (pd.Timestamp.now('Asia/Shanghai') - data_end).total_seconds() / 3600
        print(f" !!! WARNING: Using cached data from {data_start} to {data_end}")
        print(f" !!! Data is approximately {mins_behind:.1f} hours behind current time.")
        return df


# ============================================================================
# ChartGenerator (abstract product)
# ============================================================================

class ChartGenerator(ABC):
    @abstractmethod
    def create_plot(self, hist_df: pd.DataFrame, close_preds_df: pd.DataFrame, volume_preds_df: pd.DataFrame):
        ...


class CryptoChartGenerator(ChartGenerator):
    def __init__(self, config: Config):
        self.config = config

    def create_plot(self, hist_df, close_preds_df, volume_preds_df):
        print("Generating comprehensive forecast chart...")
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(15, 10), sharex=True,
            gridspec_kw={'height_ratios': [3, 1]}
        )
    
        hist_time = hist_df['timestamps']
        last_hist_time = hist_time.iloc[-1]
        pred_time = pd.to_datetime(
            [last_hist_time + timedelta(hours=i + 1) for i in range(len(close_preds_df))]
        )

        ax1.plot(hist_time, hist_df['close'], color='royalblue', label='Historical Price', linewidth=1.5)
        mean_preds = close_preds_df.mean(axis=1)
        ax1.plot(pred_time, mean_preds, color='darkorange', linestyle='-', label='Mean Forecast')
        ax1.fill_between(pred_time, close_preds_df.min(axis=1), close_preds_df.max(axis=1),
                         color='darkorange', alpha=0.2, label='Forecast Range (Min-Max)')
        ax1.set_title(
            f'{self.config.SYMBOL} Probabilistic Price & Volume Forecast (Next {self.config.PRED_HORIZON} Hours)',
            fontsize=16, weight='bold'
        )
        ax1.set_ylabel('Price (USDT)')
        ax1.legend()
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        ax2.bar(hist_time, hist_df['volume'], color='skyblue', label='Historical Volume', width=0.03)
        ax2.bar(pred_time, volume_preds_df.mean(axis=1), color='sandybrown',
                label='Mean Forecasted Volume', width=0.03)
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Time (UTC)')
        ax2.legend()
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

        separator_time = hist_time.iloc[-1] + timedelta(minutes=30)
        for ax in [ax1, ax2]:
            ax.axvline(x=separator_time, color='red', linestyle='--', linewidth=1.5, label='_nolegend_')
            ax.tick_params(axis='x', rotation=30)

        fig.tight_layout()
        chart_path = self.config.REPO_PATH / 'prediction_chart.png'
        fig.savefig(chart_path, dpi=120)
        plt.close(fig)
        print(f"Chart saved to: {chart_path}")


class StockChartGenerator(ChartGenerator):
    def __init__(self, config: Config, calendar: StockCalendar):
        self.config = config
        self.calendar = calendar

    def create_plot(self, hist_df, close_preds_df, volume_preds_df):
        print("Generating stock comprehensive forecast chart...")
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(15, 10), sharex=True,
            gridspec_kw={'height_ratios': [3, 1]}
        )

        hist_df = hist_df[hist_df['timestamps'].apply(StockCalendar.in_trading_hours)].reset_index(drop=True)
        hist_time = hist_df['timestamps']
        last_hist_time = hist_time.iloc[-1]
        step = self.config.interval_step()

        pred_time = []
        t = last_hist_time
        for _ in range(len(close_preds_df)):
            t = self.calendar.next_bar(t, step)
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
        title = f'{self.config.SYMBOL} A-Share Probabilistic Price & Volume Forecast'
        ax1.set_title(title, fontsize=16, weight='bold')
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
        chart_path = self.config.REPO_PATH / 'prediction_chart.png'
        fig.savefig(chart_path, dpi=120)
        plt.close(fig)
        print(f"Chart saved to: {chart_path}")


# ============================================================================
# MarketFactory (abstract factory)
# ============================================================================

class MarketFactory(ABC):
    @abstractmethod
    def create_config(self) -> Config:
        ...

    @abstractmethod
    def create_calendar(self) -> Calendar:
        ...

    @abstractmethod
    def create_data_fetcher(self) -> DataFetcher:
        ...

    @abstractmethod
    def create_chart_generator(self, config: Config) -> ChartGenerator:
        ...


class CryptoMarketFactory(MarketFactory):
    def create_config(self) -> Config:
        return Config(
            REPO_PATH=Path("./examples/demo"),
            MODEL_PATH="./examples/demo/models",
            SYMBOL='BTCUSDT',
            INTERVAL='1h',
            HIST_POINTS=360,
            PRED_HORIZON=24,
            N_PREDICTIONS=30,
            VOL_WINDOW=24,
        )

    def create_calendar(self) -> Calendar:
        return CryptoCalendar()

    def create_data_fetcher(self) -> DataFetcher:
        return CryptoDataFetcher()

    def create_chart_generator(self, config: Config) -> ChartGenerator:
        return CryptoChartGenerator(config)


class StockMarketFactory(MarketFactory):
    def create_config(self) -> Config:
        return Config(
            REPO_PATH=Path("./examples/demo"),
            MODEL_PATH="./examples/demo/models",
            SYMBOL='SH601600',
            INTERVAL='5min',
            HIST_POINTS=400,
            PRED_HORIZON=50,
            N_PREDICTIONS=50,
            VOL_WINDOW=50,
        )

    def create_calendar(self) -> Calendar:
        return StockCalendar()

    def create_data_fetcher(self) -> DataFetcher:
        return StockDataFetcher()

    def create_chart_generator(self, config: Config) -> ChartGenerator:
        calendar = self.create_calendar()
        return StockChartGenerator(config, calendar)


# ============================================================================
# TrendModel (uses Calendar instead of is_stock branching)
# ============================================================================

class TrendModel:
    def __init__(self, config: Config, calendar: Calendar):
        self.config = config
        self.calendar = calendar
        self.predictor = None

    def load(self):
        print("Loading Kronos model...")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k", cache_dir=self.config.MODEL_PATH)
        model = Kronos.from_pretrained("NeoQuasar/Kronos-mini", cache_dir=self.config.MODEL_PATH)

        tokenizer.eval()
        model.eval()
        self.predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
        print("Model loaded successfully.")
        return self.predictor

    def predict(self, df: pd.DataFrame):
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        last_timestamp = df['timestamps'].max()
        print(f"hist last_timestamp: {last_timestamp}")
        step = self.config.interval_step()

        new_timestamps = []
        t = last_timestamp
        while len(new_timestamps) < self.config.PRED_HORIZON:
            t = self.calendar.next_bar(t, step)
            new_timestamps.append(t)
        new_timestamps_index = pd.DatetimeIndex(new_timestamps)

        y_timestamp = pd.Series(new_timestamps_index, name='y_timestamp')
        x_timestamp = df['timestamps']
        x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]

        with torch.no_grad():
            print("Making main prediction (T=1.0)...")
            begin_time = time.time()
            pred_df = self.predictor.predict(
                df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                pred_len=self.config.PRED_HORIZON, T=1.0, top_p=0.95,
                sample_count=self.config.N_PREDICTIONS, verbose=True
            )
            print(f"Main prediction completed in {time.time() - begin_time:.2f} seconds.")
            pred_df.to_csv(
                self.config.REPO_PATH / f"{self.config.SYMBOL}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                index=True
            )
            close_preds_main = pred_df['close']
            if isinstance(close_preds_main, pd.Series):
                close_preds_main = close_preds_main.to_frame()
            volume_preds_main = pred_df['volume']
            if isinstance(volume_preds_main, pd.Series):
                volume_preds_main = volume_preds_main.to_frame()
            close_preds_volatility = close_preds_main

        return close_preds_main, volume_preds_main, close_preds_volatility


# ============================================================================
# MetricsCalculator (market-agnostic, unchanged)
# ============================================================================

class MetricsCalculator:
    def __init__(self, config: Config):
        self.config = config

    def calculate(self, hist_df, close_preds_df, v_close_preds_df):
        last_close = hist_df['close'].iloc[-1]
        final_hour_preds = close_preds_df.iloc[-1]
        upside_prob = (final_hour_preds > last_close).mean()
        print(f"Upside Probability (24h): {upside_prob:.2%}")

        hist_log_returns = np.log(hist_df['close'] / hist_df['close'].shift(1))
        historical_vol = hist_log_returns.iloc[-self.config.VOL_WINDOW:].std()
        amplification_count = 0
        for col in v_close_preds_df.columns:
            full_sequence = pd.concat([pd.Series([last_close]), v_close_preds_df[col]]).reset_index(drop=True)
            pred_log_returns = np.log(full_sequence / full_sequence.shift(1))
            predicted_vol = pred_log_returns.std()
            if predicted_vol > historical_vol:
                amplification_count += 1

        vol_amp_prob = amplification_count / len(v_close_preds_df.columns)

        print(f"Upside Probability (24h): {upside_prob:.2%}, Volatility Amplification Probability: {vol_amp_prob:.2%}")
        return upside_prob, vol_amp_prob


# ============================================================================
# HTMLUpdater (market-agnostic, unchanged)
# ============================================================================

class HTMLUpdater:
    def __init__(self, config: Config):
        self.config = config

    def update(self, upside_prob, vol_amp_prob):
        print("Updating index.html...")
        html_path = self.config.REPO_PATH / 'index.html'
        now_utc_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        upside_prob_str = f'{upside_prob:.1%}'
        vol_amp_prob_str = f'{vol_amp_prob:.1%}'

        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()

        content = re.sub(
            r'(<strong id="update-time">).*?(</strong>)',
            lambda m: f'{m.group(1)}{now_utc_str}{m.group(2)}',
            content
        )
        content = re.sub(
            r'(<p class="metric-value" id="upside-prob">).*?(</p>)',
            lambda m: f'{m.group(1)}{upside_prob_str}{m.group(2)}',
            content
        )
        content = re.sub(
            r'(<p class="metric-value" id="vol-amp-prob">).*?(</p>)',
            lambda m: f'{m.group(1)}{vol_amp_prob_str}{m.group(2)}',
            content
        )

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("HTML file updated successfully.")


# ============================================================================
# GitOperator (market-agnostic, unchanged)
# ============================================================================

class GitOperator:
    def __init__(self, config: Config):
        self.config = config

    def commit_and_push(self, commit_message):
        print("Performing Git operations...")
        try:
            os.chdir(self.config.REPO_PATH)
            subprocess.run(['git', 'add', 'prediction_chart.png', 'index.html'],
                           check=True, capture_output=True, text=True)
            commit_result = subprocess.run(['git', 'commit', '-m', commit_message],
                                           check=True, capture_output=True, text=True)
            print(commit_result.stdout)
            push_result = subprocess.run(['git', 'push'], check=True, capture_output=True, text=True)
            print(push_result.stdout)
            print("Git push successful.")
        except subprocess.CalledProcessError as e:
            output = e.stdout if e.stdout else e.stderr
            if "nothing to commit" in output or "Your branch is up to date" in output:
                print("No new changes to commit or push.")
            else:
                print(f"A Git error occurred:\n--- STDOUT ---\n{e.stdout}\n--- STDERR ---\n{e.stderr}")


# ============================================================================
# PredictionPipeline (uses MarketFactory to build components)
# ============================================================================

class PredictionPipeline:
    def __init__(self, factory: MarketFactory):
        self.factory = factory
        self.config = factory.create_config()
        self.calendar = factory.create_calendar()
        self.fetcher = factory.create_data_fetcher()
        self.charts = factory.create_chart_generator(self.config)
        self.model = None
        self.metrics = MetricsCalculator(self.config)
        self.html = HTMLUpdater(self.config)
        # self.git = GitOperator(self.config)

    def load_model(self):
        kronos = TrendModel(self.config, self.calendar)
        kronos.load()
        self.model = kronos
        return self.model

    def run_once(self, enable_git=False):
        print("\n" + "=" * 60 + f"\nStarting update task at {datetime.now(timezone.utc)}\n" + "=" * 60)

        df_full = self.fetcher.fetch(self.config)
        df_for_model = df_full.copy()

        close_preds, volume_preds, v_close_preds = self.model.predict(df_for_model)

        hist_df_for_plot = df_for_model.tail(self.config.HIST_POINTS)
        hist_df_for_metrics = df_for_model.tail(self.config.VOL_WINDOW)

        upside_prob, vol_amp_prob = self.metrics.calculate(hist_df_for_metrics, close_preds, v_close_preds)

        self.charts.create_plot(hist_df_for_plot, close_preds, volume_preds)

        self.html.update(upside_prob, vol_amp_prob)

        # if enable_git:
        #     commit_message = f"Auto-update forecast for {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
        #     self.git.commit_and_push(commit_message)

        del df_full, df_for_model, close_preds, volume_preds, v_close_preds
        del hist_df_for_plot, hist_df_for_metrics
        gc.collect()

        print("-" * 60 + "\n--- Task completed successfully ---\n" + "-" * 60 + "\n")

    def run_scheduler(self):
        while True:
            now = datetime.now(timezone.utc)
            next_run_time = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
            sleep_seconds = (next_run_time - now).total_seconds()

            if sleep_seconds > 0:
                print(f"Current time: {now:%Y-%m-%d %H:%M:%S UTC}.")
                print(f"Next run at: {next_run_time:%Y-%m-%d %H:%M:%S UTC}. Waiting for {sleep_seconds:.0f} seconds...")
                time.sleep(sleep_seconds)

            try:
                self.run_once()
            except Exception as e:
                print(f"\n!!!!!! A critical error occurred in the main task !!!!!!!")
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                print("Retrying in 5 minutes...")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                time.sleep(300)


if __name__ == '__main__':
    # Switch factory to change market:
    #   CryptoMarketFactory() - for BTCUSDT 1h crypto prediction
    #   StockMarketFactory()  - for SH601600 5min A-share prediction
    factory = StockMarketFactory()
        
    pipeline = PredictionPipeline(factory)
    pipeline.load_model()
    pipeline.run_once()
    # pipeline.run_scheduler()

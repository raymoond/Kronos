import pandas as pd
import requests
import json
from datetime import datetime
import os
import time
import random
from enum import Enum
from typing import Optional


class DataSource(Enum):
    AKSHARE = "AKShare"
    BAOSTOCK = "Baostock"
    EASTMONEY = "东方财富"


class StockDataFetcher:
    COLUMNS = ['日期', '股票代码', '开盘价', '收盘价', '最高价', '最低价',
               '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']

    COLUMN_MAPPING_AKSHARE = {
        '日期': '日期', '开盘': '开盘价', '收盘': '收盘价', '最高': '最高价',
        '最低': '最低价', '成交量': '成交量', '成交额': '成交额', '振幅': '振幅',
        '涨跌幅': '涨跌幅', '涨跌额': '涨跌额', '换手率': '换手率'
    }

    COLUMN_MAPPING_BAOSTOCK = {
        'date': '日期', 'open': '开盘价', 'high': '最高价', 'low': '最低价',
        'close': '收盘价', 'volume': '成交量', 'amount': '成交额',
        'turn': '换手率', 'pctChg': '涨跌幅'
    }

    EASTMONEY_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
    EASTMONEY_PARAMS_TEMPLATE = {
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '101',
        'fqt': '1',
        'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
    }
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
        'Referer': 'https://quote.eastmoney.com/',
        'Accept': '*/*',
    }

    def __init__(self, stock_code: str, start_date: Optional[str] = "20240101",
                 end_date: Optional[str] = None, full_history: bool = False,
                 save_dir: Optional[str] = None):
        self.stock_code = stock_code
        self.start_date = start_date 
        self.end_date = end_date or datetime.now().strftime('%Y%m%d')
        self.full_history = full_history
        self.save_dir = save_dir or self._default_save_dir()

    def _default_save_dir(self) -> str:
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "examples", "data")

    @staticmethod
    def get_market(stock_code: str) -> str:
        if stock_code.startswith(('0', '2', '3')):
            return '0'
        return '1'

    def _parse_eastmoney_klines(self, klines: list) -> Optional[pd.DataFrame]:
        stock_data = []
        for kline in klines:
            try:
                items = kline.split(',')
                if len(items) >= 6:
                    stock_data.append({
                        '日期': items[0],
                        '股票代码': self.stock_code,
                        '开盘价': float(items[1]),
                        '收盘价': float(items[2]),
                        '最高价': float(items[3]),
                        '最低价': float(items[4]),
                        '成交量': float(items[5]),
                        '成交额': float(items[6]) if len(items) > 6 else 0,
                        '振幅': float(items[7]) if len(items) > 7 else 0,
                        '涨跌幅': float(items[8]) if len(items) > 8 else 0,
                        '涨跌额': float(items[9]) if len(items) > 9 else 0,
                        '换手率': float(items[10]) if len(items) > 10 else 0
                    })
            except (ValueError, IndexError):
                continue
        if not stock_data:
            return None
        return self._finalize_df(pd.DataFrame(stock_data))

    def _finalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        df = df.sort_index()
        if not self.full_history and self.start_date is not None:
            end = self.end_date if self.end_date is not None else self.start_date
            df = df[(df.index >= pd.Timestamp(self.start_date)) & (df.index <= pd.Timestamp(end))]
        return df

    def _parse_eastmoney_jsonp(self, response_text: str) -> Optional[dict]:
        text = response_text[4:] if response_text.startswith('/**/') else response_text
        start = text.find('(')
        end = text.rfind(')')
        if start == -1 or end == -1:
            return None
        return json.loads(text[start + 1:end])

    def _parse_eastmoney_direct(self, response_text: str) -> Optional[pd.DataFrame]:
        try:
            idx = response_text.find('"klines":[')
            if idx == -1:
                return None
            start = idx + 10
            end = response_text.find(']', start)
            klines_str = response_text[start:end]
            klines = [k.strip().strip('"') for k in klines_str.split('","') if k.strip()]
            stock_data = []
            for kline in klines:
                if not kline.strip():
                    continue
                items = kline.split(',')
                if len(items) >= 6:
                    stock_data.append({
                        '日期': items[0],
                        '股票代码': self.stock_code,
                        '开盘价': float(items[1]),
                        '收盘价': float(items[2]),
                        '最高价': float(items[3]),
                        '最低价': float(items[4]),
                        '成交量': float(items[5]),
                        '成交额': float(items[6]) if len(items) > 6 else 0,
                    })
            if not stock_data:
                return None
            df = pd.DataFrame(stock_data)
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            df = df.sort_index()
            if not self.full_history and self.start_date is not None:
                end = self.end_date if self.end_date is not None else self.start_date
                df = df[(df.index >= pd.Timestamp(self.start_date)) & (df.index <= pd.Timestamp(end))]
            return df
        except Exception:
            return None

    def fetch_from_eastmoney(self) -> Optional[pd.DataFrame]:
        try:
            market = self.get_market(self.stock_code)
            secid = f"{market}.{self.stock_code}"

            if self.full_history:
                beg, end = "19900101", datetime.now().strftime('%Y%m%d')
                lmt = "50000"
            else:
                now = datetime.now()
                beg = self.start_date or "19900101"
                end = self.end_date if self.end_date and self.end_date <= now.strftime('%Y%m%d') else now.strftime('%Y%m%d')
                lmt = "10000"

            params = {
                **self.EASTMONEY_PARAMS_TEMPLATE,
                'secid': secid,
                'beg': beg,
                'end': end,
                'lmt': lmt,
                'cb': f'jQuery{random.randint(1000000, 9999999)}_{int(time.time() * 1000)}'
            }

            time.sleep(random.uniform(1, 2))
            resp = requests.get(self.EASTMONEY_URL, params=params, headers=self.HEADERS, timeout=15)

            if resp.status_code != 200:
                return None

            try:
                data = self._parse_eastmoney_jsonp(resp.text)
            except json.JSONDecodeError:
                return self._parse_eastmoney_direct(resp.text)

            if not data or data.get('data') is None:
                return None

            klines = data['data'].get('klines', [])
            if not klines:
                return None

            return self._parse_eastmoney_klines(klines)
        except Exception as e:
            print(f"东方财富获取失败: {e}")
            return None

    def fetch_from_akshare(self) -> Optional[pd.DataFrame]:
        try:
            import akshare as ak

            if self.full_history:
                df = ak.stock_zh_a_hist(symbol=self.stock_code, period="daily", adjust="qfq")
            else:
                s = self.start_date or "19900101"
                e = self.end_date or datetime.now().strftime('%Y%m%d')
                df = ak.stock_zh_a_hist(symbol=self.stock_code, period="daily",
                                        start_date=s, end_date=e, adjust="qfq")

            if df is None or df.empty:
                return None

            mapping = {k: v for k, v in self.COLUMN_MAPPING_AKSHARE.items() if k in df.columns}
            df = df.rename(columns=mapping)
            df['股票代码'] = self.stock_code
            return self._finalize_df(df)
        except ImportError:
            return None
        except Exception as e:
            print(f"AKShare获取失败: {e}")
            return None

    def fetch_from_baostock(self) -> Optional[pd.DataFrame]:
        try:
            import baostock as bs

            bs.login()
            market = self.get_market(self.stock_code)
            full_code = f"sz.{self.stock_code}" if market == '0' else f"sh.{self.stock_code}"

            if self.full_history:
                rs = bs.query_stock_basic(code=full_code)
                if rs.error_code != '0':
                    bs.logout()
                    return None
                list_date = None
                while (rs.error_code == '0') & rs.next():
                    list_date = rs.get_row_data()[2]
                if not list_date:
                    bs.logout()
                    return None
                start_date = list_date
            else:
                start_date = self.start_date[:4] + "-" + self.start_date[4:6] + "-" + self.start_date[6:8] if self.start_date else "2000-01-01"

            end_date = (self.end_date[:4] + "-" + self.end_date[4:6] + "-" + self.end_date[6:8] if self.end_date else datetime.now().strftime('%Y-%m-%d'))
            rs = bs.query_history_k_data_plus(
                full_code, "date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start_date, end_date=end_date, frequency="d", adjustflag="2"
            )

            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            bs.logout()

            if not data_list:
                return None

            df = pd.DataFrame(data_list, columns=rs.fields)
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['date'] = pd.to_datetime(df['date'])

            df = df.rename(columns=self.COLUMN_MAPPING_BAOSTOCK)
            df['股票代码'] = self.stock_code
            df.set_index('日期', inplace=True)
            df = df.sort_index()

            if not self.full_history and self.start_date is not None:
                end = self.end_date if self.end_date is not None else self.start_date
                df = df[(df.index >= pd.Timestamp(self.start_date)) & (df.index <= pd.Timestamp(end))]

            df['涨跌额'] = df['收盘价'].diff()
            df = df.dropna()
            return df
        except ImportError:
            return None
        except Exception as e:
            print(f"Baostock获取失败: {e}")
            return None

    def fetch(self) -> pd.DataFrame:
        sources = [
            (DataSource.AKSHARE, self.fetch_from_akshare),
            (DataSource.BAOSTOCK, self.fetch_from_baostock),
            (DataSource.EASTMONEY, self.fetch_from_eastmoney),
        ]

        for source, func in sources:
            print(f"\n尝试从 {source.value} 获取数据...")
            data = func()
            if data is not None and not data.empty:
                available_years = sorted(data.index.year.unique())
                print(f"获取到的数据年份: {available_years}")
                if self.full_history or (self.start_date and self.end_date and data.index.min() <= pd.Timestamp(self.end_date) and data.index.max() >= pd.Timestamp(self.start_date)):
                    print(f"{source.value} 数据获取成功！({len(data)} 条)")
                    data.attrs['data_source'] = source.value
                    return data

        print("所有数据源均失败，使用模拟数据...")
        return None
        # return self._create_sample_data()

    def _create_sample_data(self) -> pd.DataFrame:
        import numpy as np
        np.random.seed(42)

        if self.full_history:
            list_years = {'600580': 2002, '002354': 2010, '300418': 2015, '300207': 2011}
            list_year = list_years.get(self.stock_code, 2010)
            base_prices = {'600580': 8.0, '002354': 15.0, '300418': 20.0, '300207': 12.0}
            volatility = 0.02
        else:
            list_year = int((self.start_date or "20240101")[:4]) or 2024
            base_prices = {'600580': 12.0, '002354': 5.0, '300207': 15.0}
            volatility = 0.015

        base_price = base_prices.get(self.stock_code, 10.0)
        label = f"{self.stock_code} 从 {list_year} 年" if self.full_history else f"{self.start_date}-{self.end_date}"
        print(f"生成 {label} 的模拟数据...")

        start_date = datetime(list_year, 1, 1)
        all_dates = pd.bdate_range(start=start_date, end=datetime.now(), freq='B')

        if not self.full_history and self.start_date:
            sd = pd.Timestamp(self.start_date)
            ed = pd.Timestamp(self.end_date or self.start_date)
            all_dates = [d for d in all_dates if sd <= pd.Timestamp(d) <= ed]

        stock_data = []
        current_price = base_price

        for i, date in enumerate(all_dates):
            if i > 0:
                daily_return = np.random.normal(0, volatility)
                if self.full_history:
                    year = date.year
                    if year <= list_year + 2:
                        daily_return += np.random.normal(0.001, 0.01)
                    elif year <= list_year + 5:
                        daily_return += np.random.normal(0.0005, 0.005)
                    else:
                        daily_return += np.random.normal(0.0002, 0.003)
                else:
                    n = len(all_dates)
                    if i < n * 0.3:
                        daily_return += 0.0005
                    elif i < n * 0.7:
                        daily_return += -0.0003
                    else:
                        daily_return += 0.0002

                current_price *= (1 + daily_return)
                price_min = base_price * (0.3 if self.full_history else 0.5)
                price_max = base_price * (10.0 if self.full_history else 2.0)
                current_price = max(price_min, min(price_max, current_price))
            else:
                current_price = base_price

            open_price = current_price * (1 + np.random.normal(0, volatility * 0.2))
            daily_range = abs(np.random.normal(volatility * 0.8, volatility * 0.3))
            high_price = max(open_price, current_price) * (1 + daily_range)
            low_price = min(open_price, current_price) * (1 - daily_range)
            close_price = current_price

            high_price = max(open_price, close_price, low_price, high_price)
            low_price = min(open_price, close_price, high_price, low_price)

            if self.full_history:
                base_volume = max(100000, 100000 + (date.year - list_year) * 50000)
                volume_variation = abs(daily_return) * 5000000 if i > 0 else 0
            else:
                base_volume = 500000
                volume_variation = abs(daily_return) * 3000000 if i > 0 else 0
            volume = int(base_volume + volume_variation + np.random.randint(-200000, 400000))
            volume = max(50000 if self.full_history else 100000, volume)

            amount = volume * close_price / 10000

            if i > 0:
                prev_close = stock_data[-1]['收盘价']
                price_change = close_price - prev_close
                pct_change = (price_change / prev_close) * 100
            else:
                price_change = 0
                pct_change = 0

            amplitude = ((high_price - low_price) / open_price) * 100
            turnover_rate = np.random.uniform(1.0, 15.0) if self.full_history else np.random.uniform(0.5, 8.0)

            stock_data.append({
                '日期': date, '股票代码': self.stock_code,
                '开盘价': round(open_price, 2), '收盘价': round(close_price, 2),
                '最高价': round(high_price, 2), '最低价': round(low_price, 2),
                '成交量': volume, '成交额': round(amount, 2),
                '振幅': round(amplitude, 2), '涨跌幅': round(pct_change, 2),
                '涨跌额': round(price_change, 2), '换手率': round(turnover_rate, 2)
            })

        df = pd.DataFrame(stock_data)
        df.set_index('日期', inplace=True)
        df.attrs['data_source'] = '模拟数据'
        print(f"已生成 {len(df)} 条模拟数据")
        return df

    def display_info(self, df: pd.DataFrame):
        if df is None or df.empty:
            print("没有数据可显示")
            return

        source = df.attrs.get('data_source', '未知来源')
        label = f"{self.stock_code} 全部历史数据" if self.full_history else f"{self.stock_code} {self.start_date}-{self.end_date} 数据"

        print(f"\n{'=' * 60}")
        print(f"股票 {label} 摘要")
        print(f"{'=' * 60}")
        print(f"数据时间范围: {df.index.min().strftime('%Y-%m-%d')} 到 {df.index.max().strftime('%Y-%m-%d')}")
        print(f"总交易天数: {len(df):,}")
        print(f"数据来源: {source}")

        for year in sorted(df.index.year.unique()):
            yd = df[df.index.year == year]
            print(f"\n{year}年统计:")
            print(f"  交易天数: {len(yd)}")
            print(f"  平均收盘价: {yd['收盘价'].mean():.2f} 元")
            print(f"  最高价: {yd['最高价'].max():.2f} 元")
            print(f"  最低价: {yd['最低价'].min():.2f} 元")
            if len(yd) > 1:
                ret = (yd['收盘价'].iloc[-1] / yd['收盘价'].iloc[0] - 1) * 100
                print(f"  年度涨跌幅: {ret:+.2f}%")

        if self.full_history:
            total_ret = (df['收盘价'].iloc[-1] / df['收盘价'].iloc[0] - 1) * 100
            print(f"\n整体统计:")
            print(f"  总涨跌幅: {total_ret:+.2f}%")
            print(f"  历史最高价: {df['最高价'].max():.2f} 元")
            print(f"  历史最低价: {df['最低价'].min():.2f} 元")
            print(f"  平均日成交量: {df['成交量'].mean():,.0f} 股")

        latest = df.index.max()
        print(f"\n最新交易日 ({latest.strftime('%Y-%m-%d')}) 数据:")
        for col, val in df.loc[latest].items():
            if col == '股票代码':
                continue
            if col == '成交量':
                print(f"  {col}: {val:,.0f}")
            elif col == '成交额':
                print(f"  {col}: {val:,.2f} 万元")
            else:
                print(f"  {col}: {val}")

    def save(self, df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return False
        os.makedirs(self.save_dir, exist_ok=True)

        if self.full_history:
            filename = f"{self.stock_code}_all_history.csv"
        else:
            filename = f"{self.stock_code}_stock_data.csv"

        filepath = os.path.join(self.save_dir, filename)
        df.reset_index().to_csv(filepath, encoding='utf-8-sig', index=False)
        print(f"数据已保存: {filepath}")

        if self.full_history:
            years = df.index.year.unique()
            for year in years:
                yd = df[df.index.year == year].reset_index()
                yf = os.path.join(self.save_dir, f"{self.stock_code}_{year}.csv")
                yd.to_csv(yf, encoding='utf-8-sig', index=False)
            print(f"同时保存了 {len(years)} 个年份的单独文件")

        size = os.path.getsize(filepath) / 1024
        print(f"文件大小: {size:.1f} KB")
        return True

    def run(self) -> pd.DataFrame:
        label = "全部历史数据" if self.full_history else f"{self.start_date}-{self.end_date} 数据"
        print(f"{'=' * 60}")
        print(f"开始获取股票 {self.stock_code} 的 {label}")
        print(f"{'=' * 60}")
        print(f"保存目录: {self.save_dir}")

        df = self.fetch()
        if df is not None:
            self.display_info(df)
            self.save(df)
            print(f"\n股票 {self.stock_code} 数据处理完成!")
            print(f"数据时间跨度: {df.index.min().strftime('%Y-%m-%d')} 到 {df.index.max().strftime('%Y-%m-%d')}")
            print(f"总交易天数: {len(df):,}")
        else:
            print("未能获取股票数据")
        return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="股票数据获取工具")
    parser.add_argument("--stock_code", default="300418", help="股票代码")
    parser.add_argument("--start_date", default="20240101", help="开始日期 (YYYYMMDD)")
    parser.add_argument("--end_date", default=datetime.now().strftime('%Y%m%d'), help="结束日期 (YYYYMMDD)，默认当天")
    parser.add_argument("--full_history", action="store_true", help="获取全部历史数据")
    parser.add_argument("--save_dir", default="./examples/data", help="保存目录")
    args = parser.parse_args()

    fetcher = StockDataFetcher(
        stock_code=args.stock_code,
        start_date=args.start_date,
        end_date=args.end_date,
        full_history=args.full_history,
        save_dir=args.save_dir
    )
    fetcher.run()

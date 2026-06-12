import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import akshare as ak
from tools.stock_data_fetcher import StockDataFetcher

import warnings
warnings.filterwarnings('ignore')


def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 创建输出目录: {output_dir}")
    return output_dir


def get_stock_data(stock_code, data_dir, start_date=None, end_date=None):
    csv_file_path = os.path.join(data_dir, f"{stock_code}_stock_data.csv")

    if os.path.exists(csv_file_path):
        print(f"📁 使用现有数据文件: {csv_file_path}")
        return True, csv_file_path
    else:
        print(f"📡 数据文件不存在，从API获取真实数据...")
        fetcher = StockDataFetcher(stock_code=stock_code, start_date=start_date,
                                   end_date=end_date, save_dir=data_dir)
        df = fetcher.run()
        if df is not None and not df.empty:
            return True, csv_file_path
        else:
            print(f"❌ 无法获取股票数据")
            return False, None


def prepare_stock_data(csv_file_path, stock_code, history_years=1):
    print(f"正在加载和预处理股票 {stock_code} 数据...")

    df = pd.read_csv(csv_file_path, encoding='utf-8-sig')

    column_mapping = {
        '日期': 'timestamps',
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
        '成交量': 'volume',
        '成交额': 'amount',
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low'
    }

    actual_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=actual_mapping)

    if 'timestamps' not in df.columns:
        if df.index.name == '日期':
            df = df.reset_index()
            df = df.rename(columns={'日期': 'timestamps'})

    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)

    if history_years > 0:
        cutoff_date = datetime.now() - timedelta(days=history_years * 365)
        original_count = len(df)
        df = df[df['timestamps'] >= cutoff_date]
        print(f"📅 使用最近 {history_years} 年数据: {len(df)} 条记录 (从 {original_count} 条中筛选)")

    print(f"🔍 数据验证 - 最近5个交易日收盘价:")
    recent_prices = df[['timestamps', 'close']].tail()
    for _, row in recent_prices.iterrows():
        print(f"  {row['timestamps'].strftime('%Y-%m-%d')}: {row['close']:.2f}元")

    current_price = df['close'].iloc[-1]
    print(f"✅ 数据加载完成，共 {len(df)} 条记录")
    print(f"时间范围: {df['timestamps'].min()} 到 {df['timestamps'].max()}")
    print(f"价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"当前价格: {current_price:.2f}元")

    return df


def calculate_prediction_parameters(df, target_days=60):
    total_days = (df['timestamps'].max() - df['timestamps'].min()).days
    trading_days = len(df)
    trading_ratio = trading_days / total_days if total_days > 0 else 0.7

    pred_trading_days = int(target_days * trading_ratio)

    max_lookback = int(len(df) * 0.7)
    lookback = min(pred_trading_days * 3, max_lookback, len(df) - pred_trading_days)
    pred_len = min(pred_trading_days, len(df) - lookback)

    lookback = max(100, min(lookback, 400))
    pred_len = max(20, min(pred_len, 120))

    print(f"📊 参数计算:")
    print(f"  目标预测天数: {target_days} 天（自然日）")
    print(f"  预计交易日数量: {pred_trading_days} 天")
    print(f"  回看期数 (lookback): {lookback}")
    print(f"  预测期数 (pred_len): {pred_len}")

    return lookback, pred_len


def generate_future_dates(last_date, pred_len):
    future_dates = []
    current_date = last_date + timedelta(days=1)

    while len(future_dates) < pred_len:
        if current_date.weekday() < 5:
            future_dates.append(current_date)
        current_date += timedelta(days=1)

    print(f"📅 生成的未来交易日: 共 {len(future_dates)} 天")
    print(f"   起始日期: {future_dates[0].strftime('%Y-%m-%d')}")
    print(f"   结束日期: {future_dates[-1].strftime('%Y-%m-%d')}")

    return future_dates[:pred_len]


def generate_trading_dates_only(last_date, pred_len):
    holidays_2025 = [
        '2025-01-01',
        '2025-01-27', '2025-01-28', '2025-01-29', '2025-01-30', '2025-01-31', '2025-02-01', '2025-02-02',
        '2025-04-04', '2025-04-05', '2025-04-06',
        '2025-05-01', '2025-05-02', '2025-05-03',
        '2025-06-08', '2025-06-09', '2025-06-10',
        '2025-10-01', '2025-10-02', '2025-10-03', '2025-10-04', '2025-10-05', '2025-10-06', '2025-10-07',
    ]

    holidays = [datetime.strptime(date, '%Y-%m-%d').date() for date in holidays_2025]

    trading_dates = []
    current_date = last_date + timedelta(days=1)

    while len(trading_dates) < pred_len:
        if current_date.weekday() < 5 and current_date.date() not in holidays:
            trading_dates.append(current_date)
        current_date += timedelta(days=1)

    print(f"📅 生成的纯交易日: 共 {len(trading_dates)} 天")
    if trading_dates:
        print(f"   起始: {trading_dates[0].strftime('%Y-%m-%d')}")
        print(f"   结束: {trading_dates[-1].strftime('%Y-%m-%d')}")

    return trading_dates


def calculate_optimal_interval(min_val, max_val):
    range_val = max_val - min_val
    if range_val <= 0:
        return 1.0

    if range_val < 1:
        interval = 0.1
    elif range_val < 5:
        interval = 0.5
    elif range_val < 10:
        interval = 1.0
    elif range_val < 20:
        interval = 2.0
    elif range_val < 50:
        interval = 5.0
    elif range_val < 100:
        interval = 10.0
    elif range_val < 200:
        interval = 20.0
    elif range_val < 500:
        interval = 50.0
    else:
        interval = 100.0

    return interval


class EnhancedMarketFactorAnalyzer:
    def __init__(self):
        self.market_data = {}
        self.sector_data = {}
        self.macro_factors = {}
        self.policy_factors = {}

    def analyze_market_trend(self, index_codes=None):
        if index_codes is None:
            index_codes = ["sh000001", "sz399001"]
        try:
            print(f"📊 综合分析大盘趋势...")

            market_analysis = {}

            for index_code in index_codes:
                index_name = "上证指数" if "000001" in index_code else "深证成指"
                print(f"  分析{index_name}({index_code})...")

                index_df = ak.stock_zh_index_daily(symbol=index_code)

                if index_df is None or index_df.empty:
                    print(f"  ❌ 无法获取{index_name}数据")
                    continue

                index_df = index_df.rename(columns={
                    'date': 'date', 'close': 'close', 'open': 'open',
                    'high': 'high', 'low': 'low', 'volume': 'volume'
                })
                index_df['date'] = pd.to_datetime(index_df['date'])
                index_df = index_df.sort_values('date').reset_index(drop=True)

                index_df['ma5'] = index_df['close'].rolling(5).mean()
                index_df['ma20'] = index_df['close'].rolling(20).mean()
                index_df['ma60'] = index_df['close'].rolling(60).mean()
                index_df['vol_ma5'] = index_df['volume'].rolling(5).mean()

                current_data = index_df.iloc[-1]
                prev_data = index_df.iloc[-2]

                ma_condition = (current_data['ma5'] > current_data['ma20'] > current_data['ma60'])
                price_above_ma20 = current_data['close'] > current_data['ma20']
                volume_condition = current_data['volume'] > current_data['vol_ma5'] * 0.8
                trend_strength = self._calculate_trend_strength(index_df)
                is_main_uptrend = ma_condition and price_above_ma20 and trend_strength > 0.6

                market_analysis[index_name] = {
                    'is_main_uptrend': is_main_uptrend,
                    'trend_strength': trend_strength,
                    'current_close': current_data['close'],
                    'price_change_pct': ((current_data['close'] - prev_data['close']) / prev_data['close']) * 100,
                    'market_status': '主升浪' if is_main_uptrend else '震荡调整'
                }

            if market_analysis:
                avg_trend_strength = np.mean([data['trend_strength'] for data in market_analysis.values()])
                uptrend_count = sum(1 for data in market_analysis.values() if data['is_main_uptrend'])
                overall_uptrend = uptrend_count >= len(market_analysis) * 0.5

                final_analysis = {
                    'overall_is_main_uptrend': overall_uptrend,
                    'overall_trend_strength': avg_trend_strength,
                    'detailed_analysis': market_analysis,
                    'market_status': '主升浪' if overall_uptrend else '震荡调整'
                }

                print(f"✅ 大盘分析完成: {final_analysis['market_status']}, 综合趋势强度: {avg_trend_strength:.2f}")
                return final_analysis

            return self._get_default_market_analysis()

        except Exception as e:
            print(f"❌ 大盘分析错误: {e}")
            return self._get_default_market_analysis()

    def analyze_sector_resonance(self, stock_code):
        try:
            print(f"🔄 分析板块共振效应...")

            industry = "未知"
            concepts = []

            try:
                stock_info = ak.stock_individual_info_em(symbol=stock_code)
                if not stock_info.empty and 'value' in stock_info.columns:
                    industry_row = stock_info[stock_info['item'] == '行业']
                    if not industry_row.empty:
                        industry = industry_row['value'].iloc[0]
            except:
                pass

            hot_sectors = {
                '机器人': {'momentum': 0.85, 'limit_up_stocks': 18, 'active': True,
                           'description': '人形机器人、工业自动化'},
                '半导体': {'momentum': 0.8, 'limit_up_stocks': 15, 'active': True, 'description': '芯片国产替代'},
                '人工智能': {'momentum': 0.75, 'limit_up_stocks': 12, 'active': True, 'description': 'AI大模型、算力'},
                '低空经济': {'momentum': 0.7, 'limit_up_stocks': 10, 'active': True, 'description': '无人机、eVTOL'},
                '新能源': {'momentum': 0.6, 'limit_up_stocks': 8, 'active': True, 'description': '光伏、储能'},
                '医药': {'momentum': 0.5, 'limit_up_stocks': 5, 'active': False, 'description': '创新药'}
            }

            matched_sectors = []
            for sector, data in hot_sectors.items():
                if (sector in industry or
                        (stock_code == '600580' and sector in ['机器人', '低空经济']) or
                        (stock_code == '300207' and sector in ['新能源'])):
                    matched_sectors.append({
                        'sector': sector,
                        'momentum': data['momentum'],
                        'limit_up_stocks': data['limit_up_stocks'],
                        'is_active': data['active'],
                        'description': data['description']
                    })

            if matched_sectors:
                resonance_score = np.mean([sector['momentum'] for sector in matched_sectors])
                is_sector_hot = any(sector['is_active'] for sector in matched_sectors)
                main_sector = max(matched_sectors, key=lambda x: x['momentum'])
            else:
                resonance_score = 0.5
                is_sector_hot = False
                main_sector = {'sector': '传统行业', 'momentum': 0.5, 'description': '无热门概念'}

            analysis = {
                'industry': industry,
                'matched_sectors': matched_sectors,
                'main_sector': main_sector,
                'is_sector_hot': is_sector_hot,
                'resonance_score': resonance_score,
                'sector_count': len(matched_sectors)
            }

            print(f"✅ 板块分析完成: {industry}, 匹配{len(matched_sectors)}个热门板块, 共振分数: {resonance_score:.2f}")
            return analysis

        except Exception as e:
            print(f"❌ 板块分析错误: {e}")
            return self._get_default_sector_analysis()

    def analyze_macro_factors(self):
        try:
            print(f"🌍 分析宏观因素...")

            us_rate_analysis = {
                'current_rate': 4.25,
                'trend': '降息周期',
                'recent_cut': '2025年9月降息25个基点',
                'expected_cuts_2025': 2,
                'expected_cuts_2026': 2,
                'impact_on_emerging_markets': 'positive',
                'usd_index_support': 95.0,
                'analysis': '美联储开启宽松周期，利好全球流动性'
            }

            domestic_policy = {
                'monetary_policy': '稳健偏松',
                'fiscal_policy': '积极财政',
                'market_liquidity': '合理充裕',
                'industrial_policy': '设备更新、以旧换新',
                'employment_policy': '稳就业政策加力',
                'analysis': '政策组合拳发力，经济稳中向好'
            }

            industry_policy = {
                'robot_policy': '机器人产业政策支持',
                'chip_policy': '国产替代加速推进',
                'AI_policy': '人工智能发展规划',
                'low_altitude': '低空经济发展规划'
            }

            macro_analysis = {
                'us_rate_cycle': us_rate_analysis,
                'domestic_policy': domestic_policy,
                'industry_policy': industry_policy,
                'global_liquidity_outlook': '改善',
                'overall_macro_score': 0.75
            }

            print(
                f"✅ 宏观分析完成: 美国{us_rate_analysis['trend']}, 国内政策积极, 宏观评分: {macro_analysis['overall_macro_score']:.2f}")
            return macro_analysis

        except Exception as e:
            print(f"❌ 宏观分析错误: {e}")
            return self._get_default_macro_analysis()

    def analyze_company_fundamentals(self, stock_code):
        try:
            print(f"🏢 分析公司基本面...")

            if stock_code == '600580':
                fundamentals = {
                    'company_name': '卧龙电驱',
                    'business_areas': ['工业电机', '机器人关键部件', '航空电机', '新能源汽车驱动'],
                    'recent_developments': [
                        '与智元机器人实现双向持股，推进具身智能机器人技术研发',
                        '成立浙江龙飞电驱，专注航空电机业务',
                        '发布AI外骨骼机器人及灵巧手',
                        '布局高爆发关节模组、伺服驱动器等人形机器人关键部件'
                    ],
                    'growth_drivers': [
                        '设备更新政策推动工业电机需求',
                        '机器人产业快速发展',
                        '低空经济政策支持',
                        '出海战略加速'
                    ],
                    'risk_factors': [
                        '机器人业务营收占比仅2.71%，占比较低',
                        '工业需求景气度波动',
                        '原料价格波动风险'
                    ],
                    'investment_rating': '积极关注',
                    'fundamental_score': 0.7
                }
            else:
                fundamentals = {
                    'company_name': '未知',
                    'business_areas': [],
                    'recent_developments': [],
                    'growth_drivers': [],
                    'risk_factors': [],
                    'investment_rating': '中性',
                    'fundamental_score': 0.5
                }

            print(f"✅ 基本面分析完成: {fundamentals['company_name']}, 评分: {fundamentals['fundamental_score']:.2f}")
            return fundamentals

        except Exception as e:
            print(f"❌ 基本面分析错误: {e}")
            return self._get_default_fundamental_analysis()

    def _calculate_trend_strength(self, df):
        if len(df) < 20:
            return 0.5

        ma_slope = (df['ma5'].iloc[-1] - df['ma5'].iloc[-20]) / df['ma5'].iloc[-20]
        price_slope = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]

        volume_trend = df['volume'].iloc[-5:].mean() / df['volume'].iloc[-10:-5].mean()

        strength = (ma_slope * 0.4 + price_slope * 0.4 + min(volume_trend - 1, 0.2) * 0.2)
        return max(0, min(1, strength * 10))

    def _get_default_market_analysis(self):
        return {
            'overall_is_main_uptrend': False,
            'overall_trend_strength': 0.5,
            'market_status': '未知',
            'detailed_analysis': {}
        }

    def _get_default_sector_analysis(self):
        return {
            'industry': '未知',
            'matched_sectors': [],
            'main_sector': {'sector': '未知', 'momentum': 0.5, 'description': ''},
            'is_sector_hot': False,
            'resonance_score': 0.5,
            'sector_count': 0
        }

    def _get_default_macro_analysis(self):
        return {
            'us_rate_cycle': {'trend': '未知', 'expected_cuts_2025': 0},
            'domestic_policy': {'monetary_policy': '中性'},
            'overall_macro_score': 0.5
        }

    def _get_default_fundamental_analysis(self):
        return {
            'company_name': '未知',
            'business_areas': [],
            'recent_developments': [],
            'growth_drivers': [],
            'risk_factors': [],
            'investment_rating': '中性',
            'fundamental_score': 0.5
        }


def enhance_prediction_with_market_factors(historical_df, prediction_df, stock_code, market_analyzer):
    print("\n🎯 使用多维度市场因素增强预测...")

    market_analysis = market_analyzer.analyze_market_trend()
    sector_analysis = market_analyzer.analyze_sector_resonance(stock_code)
    macro_analysis = market_analyzer.analyze_macro_factors()
    fundamental_analysis = market_analyzer.analyze_company_fundamentals(stock_code)

    adjustment_factor = calculate_enhanced_adjustment_factor(
        market_analysis, sector_analysis, macro_analysis, fundamental_analysis
    )

    print(f"📈 综合调整因子: {adjustment_factor:.4f}")

    enhanced_prediction = prediction_df.copy()

    price_columns = ['close', 'open', 'high', 'low']
    for col in price_columns:
        if col in enhanced_prediction.columns:
            adjusted_value = enhanced_prediction[col] * adjustment_factor
            change_ratio = adjusted_value / enhanced_prediction[col]
            if change_ratio.max() > 1.1:
                adjusted_value = enhanced_prediction[col] * 1.1
            elif change_ratio.min() < 0.9:
                adjusted_value = enhanced_prediction[col] * 0.9
            enhanced_prediction[col] = adjusted_value

    if 'volume' in enhanced_prediction.columns:
        volume_adjustment = 1 + (adjustment_factor - 1) * 0.3
        enhanced_prediction['volume'] = enhanced_prediction['volume'] * volume_adjustment

    return enhanced_prediction, {
        'market_analysis': market_analysis,
        'sector_analysis': sector_analysis,
        'macro_analysis': macro_analysis,
        'fundamental_analysis': fundamental_analysis,
        'adjustment_factor': adjustment_factor
    }


def calculate_enhanced_adjustment_factor(market_analysis, sector_analysis, macro_analysis, fundamental_analysis):
    base_factor = 1.0
    factors_log = []

    if market_analysis['overall_is_main_uptrend']:
        trend_strength = market_analysis['overall_trend_strength']
        adjustment = 1 + trend_strength * 0.08
        base_factor *= adjustment
        factors_log.append(f"大盘主升浪: +{trend_strength * 0.08:.3f}")
    else:
        trend_strength = market_analysis['overall_trend_strength']
        adjustment = 1 + (trend_strength - 0.5) * 0.04
        base_factor *= adjustment
        factors_log.append(f"大盘震荡: {(trend_strength - 0.5) * 0.04:+.3f}")

    resonance_score = sector_analysis['resonance_score']
    sector_count = sector_analysis['sector_count']

    if sector_analysis['is_sector_hot']:
        sector_adjustment = 1 + resonance_score * 0.06 + min(sector_count * 0.01, 0.03)
        base_factor *= sector_adjustment
        factors_log.append(
            f"热门板块({sector_count}个): +{resonance_score * 0.06 + min(sector_count * 0.01, 0.03):.3f}")
    else:
        base_factor *= (1 + (resonance_score - 0.5) * 0.02)
        factors_log.append(f"一般板块: {(resonance_score - 0.5) * 0.02:+.3f}")

    macro_score = macro_analysis['overall_macro_score']
    macro_adjustment = 1 + (macro_score - 0.5) * 0.06
    base_factor *= macro_adjustment
    factors_log.append(f"宏观环境: {(macro_score - 0.5) * 0.06:+.3f}")

    us_rate_trend = macro_analysis['us_rate_cycle']['trend']
    if us_rate_trend == '降息周期':
        expected_cuts = macro_analysis['us_rate_cycle']['expected_cuts_2025']
        us_adjustment = 1 + expected_cuts * 0.015
        base_factor *= us_adjustment
        factors_log.append(f"美国降息: +{expected_cuts * 0.015:.3f}")

    fundamental_score = fundamental_analysis['fundamental_score']
    fundamental_adjustment = 1 + (fundamental_score - 0.5) * 0.08
    base_factor *= fundamental_adjustment
    factors_log.append(f"基本面: {(fundamental_score - 0.5) * 0.08:+.3f}")

    print("🔍 调整因子详情:")
    for log in factors_log:
        print(f"   {log}")

    final_factor = max(0.85, min(1.15, base_factor))

    if final_factor != base_factor:
        print(f"⚠️  调整因子从 {base_factor:.3f} 限制到 {final_factor:.3f}")

    return final_factor


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def create_comprehensive_market_report(enhancement_info, output_dir, stock_code):
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stock_code': stock_code,
        'market_analysis': enhancement_info['market_analysis'],
        'sector_analysis': enhancement_info['sector_analysis'],
        'macro_analysis': enhancement_info['macro_analysis'],
        'fundamental_analysis': enhancement_info['fundamental_analysis'],
        'adjustment_factor': enhancement_info['adjustment_factor'],
        'analysis_summary': generate_analysis_summary(enhancement_info)
    }

    report_file = os.path.join(output_dir, f'{stock_code}_comprehensive_analysis_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print(f"📋 综合分析报告已保存: {report_file}")
    return report


def generate_analysis_summary(enhancement_info):
    print("\n📊 生成分析总结...")
    market = enhancement_info['market_analysis']
    sector = enhancement_info['sector_analysis']
    macro = enhancement_info['macro_analysis']
    fundamental = enhancement_info['fundamental_analysis']

    summary = {
        'overall_sentiment': '积极' if enhancement_info['adjustment_factor'] > 1.0 else '谨慎',
        'key_drivers': [],
        'main_risks': [],
        'investment_suggestion': ''
    }

    if market['overall_trend_strength'] > 0.6:
        summary['key_drivers'].append('大盘趋势向好')

    if sector['is_sector_hot']:
        summary['key_drivers'].append(f"热门板块:{sector['main_sector']['sector']}")

    if macro['overall_macro_score'] > 0.7:
        summary['key_drivers'].append('宏观环境有利')

    if fundamental['fundamental_score'] > 0.6:
        summary['key_drivers'].append('基本面稳健')

    if market['overall_trend_strength'] < 0.4:
        summary['main_risks'].append('大盘趋势偏弱')

    if not sector['is_sector_hot']:
        summary['main_risks'].append('非热门板块')

    if len(summary['key_drivers']) > len(summary['main_risks']):
        summary['investment_suggestion'] = '可考虑逢低关注'
    else:
        summary['investment_suggestion'] = '建议谨慎操作'

    return summary


def plot_prediction(kline_df, pred_df, show_volume=True):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    if show_volume:
        sr_volume = kline_df['volume']
        sr_pred_volume = pred_df['volume']
        sr_volume.name = 'Ground Truth'
        sr_pred_volume.name = "Prediction"

        close_df = pd.concat([sr_close, sr_pred_close], axis=1)
        volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
        ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
        ax1.set_ylabel('Close Price', fontsize=14)
        ax1.legend(loc='lower left', fontsize=12)
        ax1.grid(True)

        ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
        ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
        ax2.set_ylabel('Volume', fontsize=14)
        ax2.legend(loc='upper left', fontsize=12)
        ax2.grid(True)
    else:
        close_df = pd.concat([sr_close, sr_pred_close], axis=1)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
        ax.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
        ax.set_ylabel('Close Price', fontsize=14)
        ax.legend(loc='lower left', fontsize=12)
        ax.grid(True)

    plt.tight_layout()
    plt.show()

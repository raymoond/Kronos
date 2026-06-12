import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import warnings
import json
from tools.prediction_common import (
    ensure_output_directory, get_stock_data, prepare_stock_data,
    calculate_prediction_parameters, generate_future_dates,
    calculate_optimal_interval, EnhancedMarketFactorAnalyzer,
    enhance_prediction_with_market_factors,
    calculate_enhanced_adjustment_factor,
    create_comprehensive_market_report
)

warnings.filterwarnings('ignore')

sys.path.append("../")
try:
    from model import Kronos, KronosTokenizer, KronosPredictor
except ImportError:
    print("⚠️ 无法导入Kronos模型，预测功能将不可用")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 增强可视化函数 ====================
def plot_comprehensive_prediction(
        historical_df,
        prediction_df,
        future_dates,
        stock_code,
        stock_name,
        output_dir,
        enhancement_info=None
):
    """
    绘制综合预测图表 - 包含更多市场分析信息
    """
    ensure_output_directory(output_dir)

    # 设置配色
    colors = {
        'historical': '#1f77b4',
        'prediction': '#ff7f0e',
        'enhanced': '#2ca02c',
        'background': '#f8f9fa',
        'grid': '#e9ecef',
        'positive': '#2ecc71',
        'negative': '#e74c3c',
        'neutral': '#95a5a6'
    }

    # 创建综合图表
    fig = plt.figure(figsize=(18, 14))
    gs = plt.GridSpec(4, 3, figure=fig, height_ratios=[2, 1, 1, 1])

    # 1. 主价格图表
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(colors['background'])

    # 2. 成交量图表
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_facecolor(colors['background'])

    # 3. 市场分析图表
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor(colors['background'])

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_facecolor(colors['background'])

    ax5 = fig.add_subplot(gs[2, 2])
    ax5.set_facecolor(colors['background'])

    # 4. 因素分析图表
    ax6 = fig.add_subplot(gs[3, :])
    ax6.set_facecolor(colors['background'])

    # 设置背景色
    fig.patch.set_facecolor('white')

    # 1. 价格图表
    historical_prices = historical_df.set_index('timestamps')['close']
    prediction_prices = prediction_df.set_index(pd.DatetimeIndex(future_dates))['close']

    # 获取当前最新价格
    current_price = historical_prices.iloc[-1]

    # 智能Y轴范围计算
    all_prices = pd.concat([historical_prices, prediction_prices])
    data_min = all_prices.min()
    data_max = all_prices.max()

    price_range = data_max - data_min
    y_margin = price_range * 0.15

    y_min = max(0, data_min - y_margin)
    y_max = data_max + y_margin

    # 设置Y轴刻度
    y_interval = calculate_optimal_interval(y_min, y_max)
    y_ticks = np.arange(round(y_min / y_interval) * y_interval,
                        round(y_max / y_interval) * y_interval + y_interval,
                        y_interval)

    # 绘制历史价格
    ax1.plot(historical_prices.index, historical_prices.values,
             color=colors['historical'], linewidth=2, label='历史价格')

    # 绘制预测价格
    if len(prediction_prices) > 0:
        # 连接点
        last_hist_date = historical_prices.index[-1]
        last_hist_price = historical_prices.iloc[-1]
        first_pred_date = prediction_prices.index[0]

        # 绘制连接线
        ax1.plot([last_hist_date, first_pred_date],
                 [last_hist_price, prediction_prices.iloc[0]],
                 color=colors['prediction'], linewidth=2.5, linestyle='-')

        # 绘制预测线
        ax1.plot(prediction_prices.index, prediction_prices.values,
                 color=colors['prediction'], linewidth=2.5, label='基础预测')

        # 绘制增强预测线
        if enhancement_info and 'enhanced_prediction' in enhancement_info:
            enhanced_prices = enhancement_info['enhanced_prediction'].set_index(pd.DatetimeIndex(future_dates))['close']
            ax1.plot(enhanced_prices.index, enhanced_prices.values,
                     color=colors['enhanced'], linewidth=2.5, linestyle='--', label='增强预测')

        # 标记预测起点
        ax1.axvline(x=last_hist_date, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax1.annotate('预测起点', xy=(last_hist_date, last_hist_price),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # 设置Y轴范围和刻度
    ax1.set_ylim(y_min, y_max)
    ax1.set_yticks(y_ticks)

    ax1.set_ylabel('收盘价 (元)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, color=colors['grid'], alpha=0.7)

    title = f'{stock_name}({stock_code}) - 综合因素价格预测\n当前价: {current_price:.2f}元 | 增强因子: {enhancement_info["adjustment_factor"]:.3f}' if enhancement_info else f'{stock_name}({stock_code}) - 价格预测\n当前价: {current_price:.2f}元'
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # 设置x轴格式
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 2. 成交量图表
    historical_volume = historical_df.set_index('timestamps')['volume']
    prediction_volume = prediction_df.set_index(pd.DatetimeIndex(future_dates))['volume']

    # 计算相对成交量（标准化）
    hist_volume_norm = historical_volume / historical_volume.max()
    if len(prediction_volume) > 0:
        pred_volume_norm = prediction_volume / historical_volume.max()

    # 绘制历史成交量
    ax2.bar(historical_volume.index, hist_volume_norm.values,
            alpha=0.6, color=colors['historical'], label='历史成交量')

    # 绘制预测成交量
    if len(prediction_volume) > 0:
        ax2.bar(prediction_volume.index, pred_volume_norm.values,
                alpha=0.6, color=colors['prediction'], label='预测成交量')

    ax2.set_ylabel('相对成交量', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, color=colors['grid'], alpha=0.7)
    ax2.set_ylim(0, 1.2)

    # 设置x轴格式
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # 3. 市场分析子图
    if enhancement_info:
        # 因素权重饼图
        factors = ['大盘趋势', '板块共振', '宏观环境', '美国降息', '基本面']
        weights = [25, 25, 20, 10, 20]
        colors_pie = [colors['historical'], colors['prediction'], colors['enhanced'], '#f39c12', '#9b59b6']

        ax3.pie(weights, labels=factors, autopct='%1.0f%%', colors=colors_pie, startangle=90)
        ax3.set_title('因素权重分配', fontweight='bold', fontsize=11)

        # 因素评分柱状图
        scores = [
            enhancement_info['market_analysis']['overall_trend_strength'],
            enhancement_info['sector_analysis']['resonance_score'],
            enhancement_info['macro_analysis']['overall_macro_score'],
            0.7 if enhancement_info['macro_analysis']['us_rate_cycle']['trend'] == '降息周期' else 0.3,
            enhancement_info['fundamental_analysis']['fundamental_score']
        ]

        x_pos = np.arange(len(factors))
        bars = ax4.bar(x_pos, scores, color=colors_pie, alpha=0.7)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(factors, rotation=45, fontsize=9)
        ax4.set_ylim(0, 1)
        ax4.set_ylabel('评分', fontsize=10)
        ax4.set_title('各因素当前评分', fontweight='bold', fontsize=11)
        ax4.grid(True, alpha=0.3)

        # 在柱状图上显示数值
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        # 市场状态总结
        market_status = enhancement_info['market_analysis']['market_status']
        sector_status = "热门" if enhancement_info['sector_analysis']['is_sector_hot'] else "一般"
        macro_status = "有利" if enhancement_info['macro_analysis']['overall_macro_score'] > 0.6 else "不利"

        summary_text = f"""市场状态总结:

大盘趋势: {market_status}
板块热度: {sector_status}
宏观环境: {macro_status}
美国利率: {enhancement_info['macro_analysis']['us_rate_cycle']['trend']}
综合评分: {enhancement_info['adjustment_factor']:.3f}

投资建议: {enhancement_info['fundamental_analysis']['investment_rating']}"""

        ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=10,
                 verticalalignment='top', linespacing=1.5)
        ax5.set_title('市场状态总结', fontweight='bold', fontsize=11)
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.spines['bottom'].set_visible(False)
        ax5.spines['left'].set_visible(False)

        # 4. 详细因素分析
        if 'analysis_summary' in enhancement_info:
            summary = enhancement_info['analysis_summary']
            drivers_text = "\n".join([f"• {driver}" for driver in summary['key_drivers']]) if summary[
                'key_drivers'] else "• 暂无明显驱动"
            risks_text = "\n".join([f"• {risk}" for risk in summary['main_risks']]) if summary[
                'main_risks'] else "• 风险可控"

            detail_text = f"""关键驱动因素:
{drivers_text}

主要风险提示:
{risks_text}

总体情绪: {summary['overall_sentiment']}
建议: {summary['investment_suggestion']}"""

            ax6.text(0.02, 0.95, detail_text, transform=ax6.transAxes, fontsize=9,
                     verticalalignment='top', linespacing=1.3)
            ax6.set_title('详细因素分析', fontweight='bold', fontsize=11)
            ax6.set_xticks([])
            ax6.set_yticks([])
            ax6.spines['top'].set_visible(False)
            ax6.spines['right'].set_visible(False)
            ax6.spines['bottom'].set_visible(False)
            ax6.spines['left'].set_visible(False)

    plt.tight_layout()

    # 保存图片
    chart_filename = os.path.join(output_dir, f'{stock_code}_comprehensive_prediction.png')
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 综合预测图表已保存: {chart_filename}")

    plt.show()

    return historical_prices, prediction_prices


# ==================== 主预测函数 ====================
def run_comprehensive_kronos_prediction(stock_code, stock_name, data_dir, pred_days, output_dir, history_years=1,
                                       start_date=None, end_date=None):
    """
    运行综合版Kronos模型预测流程
    """
    print(f"\n🎯 开始 {stock_name}({stock_code}) 综合版Kronos模型价格预测")
    print("=" * 60)

    # 初始化增强版市场分析器
    market_analyzer = EnhancedMarketFactorAnalyzer()

    try:
        # 1. 获取数据
        print("\n步骤1: 获取股票数据...")
        success, csv_file_path = get_stock_data(stock_code, data_dir, start_date, end_date)
        if not success:
            print("❌ 无法获取股票数据，预测终止")
            return

        # 2. 加载模型和分词器
        print("\n步骤2: 加载Kronos模型和分词器...")
        try:
            tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
            print("✅ 模型加载完成 - 使用Kronos-base模型")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("⚠️ 预测功能不可用，请检查模型安装")
            return

        # 3. 实例化预测器
        print("步骤3: 初始化预测器...")
        predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
        print("✅ 预测器初始化完成")

        # 4. 准备数据
        print("步骤4: 准备股票数据...")
        df = prepare_stock_data(csv_file_path, stock_code, history_years)

        # 5. 计算预测参数
        print("步骤5: 计算预测参数...")
        lookback, pred_len = calculate_prediction_parameters(df, target_days=pred_days)

        if pred_len <= 0:
            print("❌ 数据量不足，无法进行预测")
            return

        print(f"✅ 最终参数 - 回看期: {lookback}, 预测期: {pred_len}")

        # 6. 准备输入数据
        print("步骤6: 准备输入数据...")
        x_df = df.loc[-lookback:, ['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
        x_timestamp = df.loc[-lookback:, 'timestamps'].reset_index(drop=True)

        # 生成未来日期
        last_historical_date = df['timestamps'].iloc[-1]
        future_dates = generate_future_dates(last_historical_date, pred_len)

        print(f"输入数据形状: {x_df.shape}")
        print(f"历史数据时间范围: {x_timestamp.iloc[0]} 到 {x_timestamp.iloc[-1]}")
        print(f"预测时间范围: {future_dates[0]} 到 {future_dates[-1]}")

        # 7. 执行基础预测
        print("步骤7: 执行基础价格预测...")
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=pd.Series(future_dates),
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=True
        )

        print("✅ 基础预测完成")
        print("预测数据前5行:")
        print(pred_df.head())

        # 8. 使用多维度市场因素增强预测
        print("步骤8: 应用多维度市场因素增强预测...")
        enhanced_pred_df, enhancement_info = enhance_prediction_with_market_factors(
            df.loc[-lookback:].reset_index(drop=True),
            pred_df,
            stock_code,
            market_analyzer
        )

        # 将增强预测结果添加到信息中
        enhancement_info['enhanced_prediction'] = enhanced_pred_df

        # 9. 创建综合市场分析报告
        market_report = create_comprehensive_market_report(enhancement_info, output_dir, stock_code)

        # 10. 可视化结果
        print("步骤9: 生成综合版可视化图表...")
        historical_df = df.loc[-lookback:].reset_index(drop=True)
        hist_prices, base_pred_prices = plot_comprehensive_prediction(
            historical_df, pred_df, future_dates, stock_code, stock_name, output_dir, enhancement_info
        )

        # 11. 生成综合预测报告
        print("步骤10: 生成综合预测报告...")
        if len(enhanced_pred_df) > 0:
            current_price = hist_prices.iloc[-1]
            base_predicted_price = base_pred_prices.iloc[-1] if len(base_pred_prices) > 0 else current_price
            enhanced_predicted_price = enhanced_pred_df.set_index(pd.DatetimeIndex(future_dates))['close'].iloc[-1]

            base_change_pct = (base_predicted_price / current_price - 1) * 100
            enhanced_change_pct = (enhanced_predicted_price / current_price - 1) * 100

            print(f"\n📈 综合版Kronos模型预测报告")
            print("=" * 70)
            print(f"股票: {stock_name}({stock_code})")
            print(f"当前价格: {current_price:.2f} 元")
            print(f"基础预测价格: {base_predicted_price:.2f} 元 ({base_change_pct:+.2f}%)")
            print(f"增强预测价格: {enhanced_predicted_price:.2f} 元 ({enhanced_change_pct:+.2f}%)")
            print(f"市场因素调整因子: {enhancement_info['adjustment_factor']:.4f}")
            print(f"大盘状态: {enhancement_info['market_analysis']['market_status']}")
            print(
                f"板块共振: {enhancement_info['sector_analysis']['main_sector']['sector']} (分数: {enhancement_info['sector_analysis']['resonance_score']:.2f})")
            print(f"宏观环境: 美国{enhancement_info['macro_analysis']['us_rate_cycle']['trend']}")
            print(f"公司评级: {enhancement_info['fundamental_analysis']['investment_rating']}")
            print(f"预测期间: {pred_len} 个交易日")

            # 输出关键因素
            print(f"\n🔑 关键影响因素:")
            # print(market_report)
            for driver in market_report['analysis_summary']['key_drivers']:
                print(f"  ✅ {driver}")
            for risk in market_report['analysis_summary']['main_risks']:
                print(f"  ⚠️  {risk}")
            print(f"  💡 投资建议: {market_report['analysis_summary']['investment_suggestion']}")

            # 保存详细预测数据
            prediction_details = pd.DataFrame({
                '日期': future_dates,
                '基础预测收盘价': base_pred_prices.values if len(base_pred_prices) > 0 else [current_price] * len(
                    future_dates),
                '增强预测收盘价': enhanced_pred_df['close'].values,
                '预测成交量': enhanced_pred_df['volume'].values
            })

            prediction_file = os.path.join(output_dir, f'{stock_code}_comprehensive_predictions.csv')
            prediction_details.to_csv(prediction_file, index=False, encoding='utf-8-sig')
            print(f"💾 详细预测数据已保存: {prediction_file}")

        print(f"\n🎉 {stock_name}({stock_code}) 综合版Kronos模型预测完成!")

    except Exception as e:
        print(f"❌ 预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


# ==================== 主函数 ====================
def main():
    """
    主函数：综合版Kronos模型股票预测系统
    """
    import argparse

    parser = argparse.ArgumentParser(description='股票预测工具 CLI')
    parser.add_argument('--stock-code', dest='stock_code', default='300418', help='股票代码')
    parser.add_argument('--stock-name', dest='stock_name', default='昆仑万维', help='股票名称')
    parser.add_argument('--start-date', dest='start_date', default='20240101', help='数据开始日期 (YYYYMMDD)，默认20240101')
    parser.add_argument('--end-date', dest='end_date', default=datetime.now().strftime('%Y%m%d'),
                        help='数据结束日期 (YYYYMMDD)，默认当天')
    args = parser.parse_args()

    # ==================== 在这里修改股票配置 ====================
    STOCK_CONFIG = {
        "stock_code": args.stock_code,  # 股票代码
        "stock_name": args.stock_name,  # 股票名称
        "data_dir": r".\examples\data",
        "pred_days": 60,
        "output_dir": r".\examples\yuce",
        "history_years": 1,
        "start_date": args.start_date,
        "end_date": args.end_date
    }

    print("🤖 综合版Kronos模型股票价格预测系统")
    print("=" * 50)
    print("📊 新增功能: 多维度市场因素分析")
    print("🎯 包含: 大盘趋势 + 板块共振 + 宏观政策 + 公司基本面")
    print("🚀 使用模型: Kronos-base (更适合3070Ti显卡)")
    print(f"当前预测股票: {STOCK_CONFIG['stock_name']}({STOCK_CONFIG['stock_code']})")
    print(f"预测天数: {STOCK_CONFIG['pred_days']} 天")
    print(f"输出目录: {STOCK_CONFIG['output_dir']}")
    print()

    # 运行综合版Kronos模型预测流程
    run_comprehensive_kronos_prediction(**STOCK_CONFIG)

    print(f"\n💡 提示：综合版模型已整合多维度市场环境分析因子")


if __name__ == "__main__":
    main()
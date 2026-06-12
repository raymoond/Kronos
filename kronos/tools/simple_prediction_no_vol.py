import pandas as pd
import sys
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor
from prediction_common import plot_prediction


# 1. Load Model and Tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 2. Instantiate Predictor
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

# 3. Prepare Data
df = pd.read_csv("./examples/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 120

x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

# 4. Make Prediction
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

# 5. Visualize Results
print("Forecasted Data Head:")
print(pred_df.head())

# Combine historical and forecasted data for plotting
kline_df = df.loc[:lookback+pred_len-1]

# visualize
plot_prediction(kline_df, pred_df, show_volume=False)

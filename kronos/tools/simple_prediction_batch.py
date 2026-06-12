import pandas as pd
import sys
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor
from prediction_common import plot_prediction


# 1. Load Model and Tokenizer
TOKENIZER_PRETRAINED = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_PRETRAINED = "NeoQuasar/Kronos-base"

tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PRETRAINED)
model = Kronos.from_pretrained(MODEL_PRETRAINED)
# 2. Instantiate Predictor
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

# 3. Prepare Data
df = pd.read_csv("./examples/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 120

dfs = []
xtsp = []
ytsp = []
for i in range(5):
    idf = df.loc[(i*400):(i*400+lookback-1), ['open', 'high', 'low', 'close', 'volume', 'amount']]
    i_x_timestamp = df.loc[(i*400):(i*400+lookback-1), 'timestamps']
    i_y_timestamp = df.loc[(i*400+lookback):(i*400+lookback+pred_len-1), 'timestamps']

    dfs.append(idf)
    xtsp.append(i_x_timestamp)
    ytsp.append(i_y_timestamp)

pred_df = predictor.predict_batch(
    df_list=dfs,
    x_timestamp_list=xtsp,
    y_timestamp_list=ytsp,
    pred_len=pred_len,
)

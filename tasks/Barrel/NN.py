# %%
import pandas as pd
import numpy as np
import keras
from keras import layers
import mplhep as hep
import importlib
import NNplots as NNplots


#! SCRIVI UN IF SAVE LOAD

importlib.reload(NNplots)
hep.style.use("CMS")
df=pd.read_parquet("CCTk_match.parquet")

dropout=0.1
val_split=0.2
epochs=30
batch_size=512

y=df["CryClu_label"].to_numpy()
#genpt=df["CryClu_genPt"].to_numpy()
#evIdx=df["CryClu_evIdx"].to_numpy()

df=df.drop(columns=["CryClu_label",
                    "CryClu_genPt",
                    "CryClu_evIdx"])

df = (df - df.mean()) / df.std()

#%%
#keras mlp
model = keras.Sequential(
    [
        layers.Dense(64, activation="swish", input_shape=[len(df.keys())]),
        layers.Dropout(dropout),
        layers.Dense(64, activation="swish"),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid"),
    ]
)

#keras compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

weights1 = len(y) / (2 * np.sum(y == 1))
weights0 = len(y) / (2 * np.sum(y == 0))
class_weights = {0: weights0, 1: weights1}
history = model.fit(
    df, y,
    validation_split=val_split,
    epochs=epochs,
    batch_size=batch_size,
    class_weight=class_weights,
    verbose=2
)
#%%
df_val = pd.read_parquet("CCTk_match.parquet")[-int(val_split * len(df)):]
y_val = df_val["CryClu_label"].to_numpy().ravel()
y_pred = model.predict(df[-int(val_split * len(df)):]).ravel()


# %%

y_pred_atanh=np.arctanh(y_pred)
y_pred_atanh[y_pred_atanh==np.inf]=max(y_pred_atanh[y_pred_atanh!=np.inf])

importlib.reload(NNplots)
NNplots.plot_loss(history)
NNplots.conf_matrix(y_pred, y_val)
NNplots.roc_plot(y_pred, y_val,xlim=0.005)
NNplots.out_plot(y_pred_atanh, y_val,significance=True)
NNplots.tr_binned_eff(y_pred, y_val, df_val["CryClu_genPt"])
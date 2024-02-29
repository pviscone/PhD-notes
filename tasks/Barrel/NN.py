# %%
import pandas as pd
import numpy as np
import keras
from keras import layers
import mplhep as hep
import importlib
import NNplots as NNplots

importlib.reload(NNplots)
hep.style.use("CMS")
df = pd.read_parquet("CCTk_match.parquet")

dropout = 0.1
val_split = 0.2
epochs = 50
batch_size = 512
pt_cut=2
load = False
save = False

columns_nn = [
    "CryClu_standaloneWP",
    "CryClu_showerShape",
    "CryClu_isolation",
    "Tk_hitPattern",
    "Tk_nStubs",
    "Tk_chi2Bend",
    "Tk_chi2RPhi",
    "Tk_chi2RZ",
    "Tk_vz",
    "CCTk_dR",
    "CCTk_dPt",
]
#%%
df = df[df["CryClu_pt"] > pt_cut]
y = df["CryClu_label"].to_numpy()
evIdx = df["CryClu_evIdx"].to_numpy()
val_start_evidx = evIdx[int(len(evIdx) * (1 - val_split))]

#%%
df = df[[*columns_nn]]
df = (df - df.mean()) / df.std()

# %%
# keras mlp
if not load:
    model = keras.Sequential(
        [
            layers.Dense(64, activation="swish", input_shape=[len(df.keys())]),
            layers.Dropout(dropout),
            layers.Dense(64, activation="swish"),
            layers.Dropout(dropout),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    # keras compile
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.summary()

    weights1 = len(y) / (2 * np.sum(y == 1))
    weights0 = len(y) / (2 * np.sum(y == 0))
    class_weights = {0: weights0, 1: weights1}
    history = model.fit(
        df,
        y,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        verbose=2,
    )
    NNplots.plot_loss(history)
    if save:
        model.save(save)
else:
    model = keras.models.load_model(load)
# %%
df_val = pd.read_parquet("CCTk_match.parquet")
df_val = df_val[df_val["CryClu_pt"] > pt_cut]

val_start_idx = np.where(
    df_val["CryClu_evIdx"] == val_start_evidx)[0][0]
df_val=df_val[val_start_idx:]
y_val = df_val["CryClu_label"].to_numpy().ravel()

nn_input_val = df_val[[*columns_nn]]
nn_input_val = (nn_input_val - nn_input_val.mean()) / nn_input_val.std()

y_pred = model.predict(nn_input_val).ravel()



# %%

y_pred_atanh = np.arctanh(y_pred)
y_pred_atanh[y_pred_atanh == np.inf] = max(y_pred_atanh[y_pred_atanh != np.inf])

importlib.reload(NNplots)
NNplots.conf_matrix(y_pred, y_val)
NNplots.roc_plot(y_pred, y_val, xlim=0.005)
NNplots.out_plot(y_pred_atanh, y_val, significance=True)
NNplots.loop_on_trs(
    NNplots.efficiency_plot,
    y_pred,
    y_val,
    df_val["CryClu_genPt"].to_numpy(),
)

# %%
import pandas as pd
import numpy as np
import keras
from keras import layers
import mplhep as hep
import importlib
import NNplots as NNplots
from matplotlib import pyplot as plt

importlib.reload(NNplots)
hep.style.use("CMS")
df = pd.read_parquet("CCTk_match.parquet")

dropout = 0.1
val_split = 0.2
epochs = 50
batch_size = 512
pt_cut_train = 0
pt_cut_test = 0
load = "NN.keras"
# load = False
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
# %%
df = df[df["CryClu_pt"] > pt_cut_train]
y = df["CryClu_label"].to_numpy()
evIdx = df["CryClu_evIdx"].to_numpy()
val_start_evidx = evIdx[int(len(evIdx) * (1 - val_split))]

# %%
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
df_val = df_val[df_val["CryClu_pt"] > pt_cut_test]

val_start_idx = np.where(df_val["CryClu_evIdx"] == val_start_evidx)[0][0]
df_val = df_val[val_start_idx:]
y_val = df_val["CryClu_label"].to_numpy().ravel()

nn_input_val = df_val[[*columns_nn]]
nn_input_val = (nn_input_val - nn_input_val.mean()) / nn_input_val.std()

y_pred = model.predict(nn_input_val).ravel()


# %%

y_pred_atanh = np.arctanh(y_pred)
y_pred_atanh[y_pred_atanh == np.inf] = max(y_pred_atanh[y_pred_atanh != np.inf])

importlib.reload(NNplots)
NNplots.conf_matrix(y_pred, y_val)
NNplots.roc_plot(y_pred, y_val)
NNplots.out_plot(y_pred_atanh, y_val, significance=True)
NNplots.loop_on_trs(
    NNplots.efficiency_plot,
    y_pred,
    y_val,
    df_val["CryClu_genPt"].to_numpy(),
)


# %%
pt_cuts = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 5, 10, 15, 20])
NNplots.roc_pt(y_pred, y_val, pt_cuts, df_val)


# %%
import awkward as ak

score, truth, pt = NNplots.pt_score_ak(
    ak.ArrayBuilder(),
    ak.ArrayBuilder(),
    ak.ArrayBuilder(),
    y_pred,
    y_val,
    df_val["CryClu_pt"].to_numpy(),
    df_val["CryClu_evIdx"].to_numpy(),
)

score = ak.to_numpy(
    score[ak.argmax(pt, axis=1, keepdims=True)], allow_missing=False
).ravel()
truth = ak.to_numpy(
    truth[ak.argmax(pt, axis=1, keepdims=True)], allow_missing=False
).ravel()
pt = ak.to_numpy(pt[ak.argmax(pt, axis=1, keepdims=True)], allow_missing=False).ravel()
# %%

fig, ax = plt.subplots()
pt_array = np.linspace(0, 40, 40)


def f(s):
    res = []
    for elem in pt_array:
        pt_mask = pt > elem
        score_mask = score > s
        mask = pt_mask & score_mask
        fraction = sum(mask) / len(mask)
        # rate
        res.append(fraction * 11245.6 * 2500 / 1e3)
    return res


for idx, s in enumerate(np.tanh(np.linspace(0.5, 6.5, 1))):
    if idx > 5:
        style = "--"
    else:
        style = "-"
    res = f(s)
    ax.plot(pt_array, res, style, label=f"score > {np.arctanh(s):.2f}")
ax.set_xlabel("$p_T $ cut [GeV]")
ax.set_ylabel("Trigger Rate [kHz]")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.grid()
# Put a legend to the right of the current axis
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

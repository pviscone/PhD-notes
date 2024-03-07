
#%%
import pandas as pd
import numpy as np
import keras
from keras import layers
import mplhep as hep
import importlib
import NNplots as NNplots
from sklearn.model_selection import train_test_split

hep.style.use("CMS")

df_name = "dataset.parquet"
dropout = 0.025
val_split = 0.2
epochs = 50
lr = 0.001
batch_size = 512
pt_cut_train = 0
pt_cut_test = 0
load = "NN.keras"
#load = False
save = False

random_seed = 666

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
df = pd.read_parquet(df_name)
np.random.seed(random_seed)
ev_idx = df["CryClu_evIdx"].unique()
np.random.shuffle(ev_idx)
df=df.groupby("CryClu_evIdx").apply(lambda x: x).loc[ev_idx].reset_index(drop=True)

y = df["CryClu_label"].to_numpy()
df_train, df_val, y_train, y_val = train_test_split(df, y, test_size=val_split)

df_train = df_train[df_train["CryClu_pt"] > pt_cut_train]
df_val = df_val[df_val["CryClu_pt"] > pt_cut_test]

mean = (df_train.groupby("CryClu_label").mean()).mean()
std = np.sqrt((df_train.groupby("CryClu_label").var().mean()))

df_train = (df_train - mean) / std
df_train = df_train[[*columns_nn]]

nn_input_val = (df_val - mean) / std
nn_input_val = nn_input_val[[*columns_nn]]
# %%
# keras mlp
if not load:
    model = keras.Sequential(
        [
            layers.Dense(64, activation="swish", input_shape=[len(df_train.keys())]),
            layers.Dropout(dropout),
            layers.Dense(64, activation="swish"),
            layers.Dropout(dropout),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    # keras compile
    optimizer = keras.optimizers.RMSprop(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    model.summary()

    weights1 = len(y_train) / (2 * np.sum(y_train == 1))
    weights0 = len(y_train) / (2 * np.sum(y_train == 0))
    class_weights = {0: weights0, 1: weights1}
    history = model.fit(
        df_train,
        y_train,
        validation_data=(nn_input_val, y_val),
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


y_pred = model.predict(nn_input_val).ravel()

y_pred_atanh = np.arctanh(y_pred)
y_pred_atanh[y_pred_atanh == np.inf] = max(y_pred_atanh[y_pred_atanh != np.inf])


# %%
importlib.reload(NNplots)

if not load:
    NNplots.conf_matrix(y_pred, y_val)

NNplots.roc_plot(y_pred, y_val, xlim=[-0.0005, 0.005])
NNplots.out_plot(y_pred_atanh, y_val, significance=False)

pt_cuts = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 5, 10, 15, 20])
NNplots.roc_pt(y_pred, y_val, pt_cuts, df_val, xlim=(0.00008, 1))


# %%
importlib.reload(NNplots)
#!! TOO SLOW
NNplots.rate_pt_plot(
    y_pred[y_val == 0][:100000],
    df_val[y_val == 0][:100000],
    log=True,
    y_trs=np.tanh(np.linspace(1, 4, 4)),
    pt_trs=np.linspace(0, 40, 20),
)

#%%

importlib.reload(NNplots)
ax_with_match=NNplots.loop_on_trs(
    NNplots.efficiency_plot,
    y_pred,
    y_val,
    df_val["CryClu_genPt"].to_numpy(),
    trs=np.array([0.1,1,2,3,4]),
    bins=np.linspace(0, 100, 31),
    matchingCC=True,
    TkEle=True,
)
ax_with_match.set_title("Class*Match eff")
ax_with_match.grid()
#%%

importlib.reload(NNplots)
ax_not_matched = NNplots.loop_on_trs(
    NNplots.efficiency_plot,
    y_pred,
    y_val,
    df_val["CryClu_genPt"].to_numpy(),
    trs=np.array([0.1,1,2,3,4]),
    bins=np.linspace(0, 100, 31),
    matchingCC=False,
    TkEle=True,
)
ax_not_matched.set_title("Classification only eff")
ax_not_matched.grid()
# %%
importlib.reload(NNplots)
ax_match=NNplots.matching_plot(bins=np.linspace(0, 100, 31))
ax_match.set_title("Match eff")
# %%

import corner
df_val["score"] = np.nan_to_num(np.arctanh(y_pred),8)

df_sig = df_val[df_val["CryClu_label"] == 1].drop(columns=["CryClu_label","CryClu_evIdx"])
df_bkg = df_val[df_val["CryClu_label"] == 0].drop(columns=["CryClu_label","CryClu_evIdx"])

df_bkg["CryClu_genPt"] = np.random.normal(-1, 1e-5, size=len(df_bkg))

labels=[key.split("_")[-1] for key in df_sig.keys()]



import matplotlib.lines as mlines
import matplotlib.pyplot as plt

bins=[
    10,
    30,
    30,
    30,
    15,
    30,
    50,
    30,
    40,
    40,
    40,
    25,
    50,
    50,
    50


]
h_range = [
    [-0.1, 1.1],
    [-0.1, 1.1],
    [-3, 25],
    [0, 70],
    [-2, 12],
    [-1, 7],
    [-7, 70],
    [-5, 15],
    [-20, 20],
    [-0.15, 0.6],
    [-100, 40],
    [-4, 20],
    [-10, 110],
    [-5, 110],
    [-1,10],
]


fig=corner.corner(df_sig, labels=labels, color="tab:blue", levels=(0.5,0.9, 0.99), scale_hist=True,bins=bins,range=h_range,plot_density=True )

corner.corner(df_bkg[:len(df_sig)], labels=labels, color="tab:orange", levels=(0.5,0.9, 0.99), scale_hist=True,bins=bins,range=h_range,plot_density=True, fig=fig)

blue_line = mlines.Line2D([], [], color="tab:blue", label="Signal")
red_line = mlines.Line2D([], [], color="tab:orange", label="Background")


plt.legend(
    handles=[blue_line, red_line],
    loc="upper right",
    frameon=False,
    bbox_to_anchor=(1, 5),
    fontsize=25,
)

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

df_name = "CCTk_match.parquet"
dropout = 0.05
val_split = 0.2
epochs = 50
lr = 0.001
batch_size = 512
pt_cut_train = 0
pt_cut_test = 0
load = "NN.keras"
#load = False
save = False

random_state = 666

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
df = pd.read_parquet(df_name).sample(frac=1, random_state=random_state)


df = df[df["CryClu_pt"] > pt_cut_train]
y = df["CryClu_label"].to_numpy()

# %%


mean = (df.groupby("CryClu_label").mean()).mean()
std = np.sqrt((df.groupby("CryClu_label").var().mean()))

df = (df - mean) / std

df = df[[*columns_nn]]
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
    optimizer = keras.optimizers.RMSprop(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

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
new_df = pd.read_parquet("CCTk_match.parquet").sample(frac=1, random_state=random_state)

new_df = new_df[new_df["CryClu_pt"] > pt_cut_test]
new_y = new_df["CryClu_label"].to_numpy().ravel()

_, df_val, _, y_val = train_test_split(new_df, new_y, test_size=val_split)


nn_input_val = (df_val - mean) / std
nn_input_val = nn_input_val[[*columns_nn]]
y_pred = model.predict(nn_input_val).ravel()

y_pred_atanh = np.arctanh(y_pred)
y_pred_atanh[y_pred_atanh == np.inf] = max(y_pred_atanh[y_pred_atanh != np.inf])


# %%
importlib.reload(NNplots)
NNplots.conf_matrix(y_pred, y_val)
NNplots.roc_plot(y_pred, y_val, xlim=[-0.0005, 0.005])
NNplots.out_plot(y_pred_atanh, y_val, significance=True)
NNplots.loop_on_trs(
    NNplots.efficiency_plot,
    y_pred,
    y_val,
    df_val["CryClu_genPt"].to_numpy(),
)

pt_cuts = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 5, 10, 15, 20])
NNplots.roc_pt(y_pred, y_val, pt_cuts, df_val)


# %%
importlib.reload(NNplots)
#!! IL RATE FALLO SOLO CON BKG
#!!TO FIX!
NNplots.rate_pt_plot(y_pred, df_val)

# %%


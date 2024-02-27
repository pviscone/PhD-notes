# %%
import pickle
import numpy as np
import keras
from keras import layers
import numba
import matplotlib.pyplot as plt
import mplhep as hep
import importlib
import NNplots

importlib.reload(NNplots)

#%%
dataset = "pickle_data.p"
load = "crossAtt.keras"  # If false it will train a new model
save = False
plots = True
NNplots.savefig = 'fig/NNplots'
#NNplots.img_format = 'pdf' #pdf default

validation_split = 0.2

hep.style.use("CMS")


def get_data(dataset):
    pkl = pickle.load(open(dataset, "rb"))
    dfDict = pkl["data"]
    varDict = pkl["columns"]
    genPt = pkl["genpt"]  # it is the genPt of the gen that match
    # the CryClu, 0 if there is no match

    Tk = dfDict["Tk"]
    CryClu = dfDict["CryClu"]
    y = CryClu[:, :, -1]
    CryClu = CryClu[:, :, :-1]
    varDict["CryClu"] = varDict["CryClu"][:-1]

    weights1 = len(y[y != -999]) / (2 * np.sum(y.ravel() == 1))
    weights0 = len(y[y != -999]) / (2 * np.sum(y.ravel() == 0))
    class_weights = {0: weights0, 1: weights1}
    y[y == -999] = 0
    return Tk, CryClu, y, genPt, varDict, class_weights


@numba.jit
def build_mask(Tk, CryClu):
    mask = np.ones((Tk.shape[0], CryClu.shape[1], Tk.shape[1]), dtype=numba.types.bool_)
    for ev in range(Tk.shape[0]):
        for c in range(CryClu.shape[1]):
            if CryClu[ev, c, 0] == -999:
                mask[ev, c, :] = False
        for t in range(Tk.shape[1]):
            if Tk[ev, t, 0] == -999:
                mask[ev, :, t] = False
    return mask


def build_model(CryClu, Tk, dropout=0.05, nunits=32, nlayers=2, activation="swish"):
    CryCluInput = keras.Input(
        shape=(
            CryClu.shape[1],
            CryClu.shape[2],
        ),
        name="CryCluInput",
    )
    xC = layers.BatchNormalization()(CryCluInput)

    for _ in range(nlayers):
        xC = layers.Dense(nunits, activation=activation)(xC)
        xC = layers.Dropout(dropout)(xC)

    TkInput = keras.Input(shape=(Tk.shape[1], Tk.shape[2]), name="TkInput")
    xT = layers.BatchNormalization()(TkInput)

    for _ in range(nlayers):
        xT = layers.Dense(nunits, activation=activation)(xT)
        xT = layers.Dropout(dropout)(xT)

    maskInput = keras.Input(shape=(CryClu.shape[1], Tk.shape[1]), name="maskInput")
    crossAtt = layers.MultiHeadAttention(
        num_heads=1, key_dim=nunits, value_dim=nunits, dropout=dropout
    )(
        xC,
        xT,
        xT,
        attention_mask=maskInput,
    )

    x = layers.Add()([crossAtt, xC])
    xadd = layers.LayerNormalization()(x)

    for i in range(nlayers):
        if i == 0:
            inputx = xadd
        else:
            inputx = x

        x = layers.Dense(nunits, activation=activation)(inputx)
        x = layers.Dropout(dropout)(x)

    x = layers.Add()([xadd, x])
    x = layers.LayerNormalization()(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=[CryCluInput, TkInput, maskInput], outputs=x)
    return model


def compile_model(
    model,
    CryClu,
    Tk,
    mask,
    learning_rate=0.001,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
):
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    history = model.fit(
        [CryClu, Tk, mask],
        y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        class_weight=class_weights,
    )
    return history


def plot_history(history, item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


Tk, CryClu, y, genPt, varDict, class_weights = get_data(dataset)
mask = build_mask(Tk, CryClu)


if not load:
    model = build_model(CryClu, Tk, dropout=0.05)

    history = compile_model(
        model,
        CryClu,
        Tk,
        mask,
        batch_size=128,
        epochs=15,
        learning_rate=0.001,
        validation_split=validation_split,
    )
    plot_history(history, "loss")
else:
    model = keras.models.load_model(load)
# %% #

if save:
    model.save(save)

# Plotta
if plots:
    val_slice = np.s_[-int(y.shape[0] * (validation_split)) :]

    genPt_test = genPt[val_slice]
    CryClu_test = CryClu[val_slice]
    Tk_test = Tk[val_slice]
    mask_test = mask[val_slice]

    if 'y_pred' not in globals():
        y_pred = model.predict([CryClu_test, Tk_test, mask_test])
    y_test = y[val_slice]

    etaIdx = idx = varDict["CryClu"].index("eta")
    phiIdx = idx = varDict["CryClu"].index("phi")
    eta = CryClu_test[:, :, etaIdx]
    phi = CryClu_test[:, :, phiIdx]

    NNplots.trigger_rate(y_pred, y_test)
    NNplots.conf_matrix(y_pred, y_test)
    NNplots.genPt_eff(genPt_test, y_pred, y_test)
    NNplots.eta_phi_eff(eta, phi, y_pred, y_test)
    NNplots.corner_TPvsFN(CryClu_test, y_pred, y_test, varDict["CryClu"])

# %%

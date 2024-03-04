import matplotlib.pyplot as plt
import mplhep as hep
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import roc_curve
import numpy as np
import awkward as ak
import numba as nb
from utils import snapshot_wrapper

hep.style.use("CMS")


def plot_loss(history, ax=False):
    if not ax:
        fig, ax = plt.subplots()
    ax.plot(history.history["loss"], label="loss")
    ax.plot(history.history["val_loss"], label="val_loss")
    ax.grid()
    ax.legend()
    return ax


def conf_matrix(y_pred, y_test, ax=False):
    if not ax:
        fig, ax = plt.subplots()
    conf = confusion_matrix(y_test.ravel(), y_pred.ravel() > 0.5, normalize="true")
    print(conf)
    im = ax.imshow(conf, cmap="Blues", interpolation="nearest")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1], ["Background", "Signal"])
    ax.set_yticks([0, 1], ["Background", "Signal"])
    return ax


# * 11245.6 * 2500 / 1e3
def roc_plot(y_pred, y_test, *args, ax=False, xlim=False, scatter=True, **kwargs):
    if not ax:
        fig, ax = plt.subplots()
    fpr, tpr, tr = roc_curve(y_test.ravel(), y_pred.ravel(), drop_intermediate=False)
    tr = np.arctanh(tr)

    if scatter:
        # plot the eddiciecy vs trigger rate for phase 2
        mappable = ax.scatter(fpr, tpr, c=tr, marker=".", *args, **kwargs)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(
            mappable, ax=ax, cax=cax, label="Threshold", orientation="vertical"
        )
    else:
        ax.plot(fpr, tpr, *args, linewidth=2, **kwargs)

    ax.set_xlabel("Bkg efficiency")
    ax.set_ylabel("Electron efficiency")
    ax.grid()
    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(-0.0005, ax.get_xlim()[1])
    return ax


def out_plot(y_pred, y_test, bins=30, significance=False, ax=False):
    if not ax:
        fig, ax = plt.subplots()
    hs = ax.hist(
        y_pred[y_test == 1],
        bins=bins,
        histtype="step",
        density=False,
        label="Signal",
        linewidth=2,
    )
    hb = ax.hist(
        y_pred[y_test == 0],
        bins=bins,
        histtype="step",
        density=False,
        label="Background",
        linewidth=2,
    )

    ax.set_yscale("log")

    if significance:
        centers = (hs[1][:-1] + hs[1][1:]) / 2
        ax.step(
            centers, hs[0] / np.sqrt(hs[0] + hb[0]), label="S/$\sqrt{S+B}$", where="mid"
        )

    ax.set_xlabel("atanh(NNScore)")
    ax.grid()
    ax.legend()
    hep.cms.text("Phase-2 Simulation")
    return ax


def efficiency_plot(y_pred, y_test, genpt, threshold=0.5, label=None, ax=False):
    if not ax:
        _, ax = plt.subplots()

    genHist = np.histogram(genpt[y_test == 1], bins=50, range=(0, 100))

    genPredHist = np.histogram(
        genpt[np.bitwise_and(y_test == 1, (y_pred > threshold).astype(int) == y_test)],
        bins=50,
        range=(0, 100),
    )

    centers = (genHist[1][:-1] + genHist[1][1:]) / 2
    ax.step(
        centers,
        genPredHist[0] / genHist[0],
        where="mid",
        label=f"Threshold: {np.arctanh(threshold):.2f}",
        linewidth=2,
    )
    ax.set_xlabel("genPt [GeV]")
    ax.set_ylabel("Efficiency")
    ax.grid()
    return ax


def loop_on_trs(func, *args, ax=False, trs=np.linspace(0.1, 3, 6), **kwargs):
    if not ax:
        _, ax = plt.subplots()
    for tr in np.tanh(trs):
        func(*args, threshold=tr, ax=ax, **kwargs)
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.grid()
    return ax


def roc_pt(y_pred, y_test, pt_cuts, df_val, xlim=(0.00008, 0.5), log=True, ax=False):
    if not ax:
        _, ax = plt.subplots()

    for idx, pt in enumerate(pt_cuts):
        if idx > 5:
            style = "--"
        else:
            style = "-"
        roc_plot(
            y_pred[df_val["CryClu_pt"] > pt],
            y_test[df_val["CryClu_pt"] > pt],
            style,
            scatter=False,
            ax=ax,
            label=f"pt > {pt:.1f} GeV",
        )
    ax.legend()
    if log:
        ax.set_xscale("log")

    if xlim:
        ax.set_xlim(xlim)
    hep.cms.text("Phase-2 Simulation")


#!Numba does not support reurn index true in unique, FIX!
#Can i pass the offset from outside???
@snapshot_wrapper
@nb.jit
def pt_score_ak(pt_builder, y_pred_builder, y_pred, pt, offset):
    ev_lenghts = offset[1:] - offset[:-1]
    counter = 0
    for lenght in ev_lenghts:
        pt_builder.begin_list()
        y_pred_builder.begin_list()
        for i in range(lenght):
            idx = counter + i
            pt_builder.append(pt[idx])
            y_pred_builder.append(y_pred[idx])
        pt_builder.end_list()
        y_pred_builder.end_list()
        counter += lenght
    return (y_pred_builder, pt_builder)


@nb.jit
def rate_pt_res(pt_trs, pt, score, s):
    res = []
    for elem in pt_trs:
        pt_mask = pt > elem
        score_mask = score > s
        mask = pt_mask & score_mask
        fraction = sum(mask) / len(mask)
        # rate
        res.append(fraction * 11245.6 * 2500 / 1e3)
    return res


def rate_pt_plot(
    y_pred,
    df,
    pt_trs=np.linspace(0, 40, 40),
    score_trs=np.tanh(np.linspace(0.5, 6.5, 6)),
    ax=False,
):
    if not ax:
        _, ax = plt.subplots()

    print("start")
    ev_idx=df["CryClu_evIdx"].to_numpy()
    _,offset = np.unique(ev_idx, return_index=True)
    offset = np.append(offset, len(ev_idx))
    score, pt = pt_score_ak(
        ak.ArrayBuilder(),
        ak.ArrayBuilder(),
        y_pred,
        df["CryClu_pt"].to_numpy(),
        offset,
    )
    print("1")
    score = ak.to_numpy(
        score[ak.argmax(pt, axis=1, keepdims=True)], allow_missing=False
    ).ravel()
    print("2")
    #!PERCHE SEMPLICEMENTE NON PRENDO MAX??????
    pt = ak.to_numpy(
        pt[ak.argmax(pt, axis=1, keepdims=True)], allow_missing=False
    ).ravel()
    print("3")
    for idx, s in enumerate(score_trs):
        if idx > 5:
            style = "--"
        else:
            style = "-"

        """
            res = []
            for elem in pt_trs:
                pt_mask = pt > elem
                score_mask = score > s
                mask = pt_mask & score_mask
                fraction = sum(mask) / len(mask)
                # rate
                res.append(fraction * 11245.6 * 2500 / 1e3)
        """
        res = rate_pt_res(pt_trs, pt, score, s)
        print(idx)
        ax.plot(pt_trs, res, style, label=f"score > {np.arctanh(s):.2f}")
    ax.set_xlabel("$p_T $ cut [GeV]")
    ax.set_ylabel("Trigger Rate [kHz]")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.grid()
    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))



import matplotlib.pyplot as plt
import mplhep as hep
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import roc_curve
import numpy as np
import awkward as ak
import numba as nb
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from utils import label_builder

hep.style.use("CMS")

root_signal = "DoubleElectrons131X.root"

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


def out_plot(y_pred, y_test, bins=30, significance=False, ax=False,density=True):
    if not ax:
        fig, ax = plt.subplots()
    hs = ax.hist(
        y_pred[y_test == 1],
        bins=bins,
        histtype="step",
        density=density,
        label="Signal",
        linewidth=2,
    )
    hb = ax.hist(
        y_pred[y_test == 0],
        bins=bins,
        histtype="step",
        density=density,
        label="Background",
        linewidth=2,
    )

    ax.set_yscale("log")

    if density:
        ax.set_ylabel("Density")

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


def efficiency_plot(y_pred,
                    y_test,
                    genpt,
                    matchingCC=False,
                    TkEle=False,
                    threshold=0.5,
                    bins=np.linspace(0,100,31),
                    ax=False):
    if not ax:
        _, ax = plt.subplots()

    genHist = np.histogram(genpt[y_test == 1], bins=bins)

    genPredHist = np.histogram(
        genpt[np.bitwise_and(y_test == 1, (y_pred > threshold).astype(int) == y_test)],
        bins=bins
    )




    if matchingCC:
        scale=get_matching_curve(bins,obj="CryClu")
    else:
        scale=1

    centers = (genHist[1][:-1] + genHist[1][1:]) / 2
    ax.step(
        centers,
        scale*genPredHist[0] / genHist[0],
        where="mid",
        label=f"Threshold: {np.arctanh(threshold):.2f}",
        linewidth=2,
    )

    if TkEle:
        tk_eff = get_matching_curve(bins, obj="TkEle")
        centers = (genHist[1][:-1] + genHist[1][1:]) / 2
        ax.step(
            centers,
            tk_eff,
            where="mid",
            label="TkEle",
            linewidth=2,
            linestyle="-.",
            color="green",
        )

    ax.set_xlabel("genPt [GeV]")
    ax.set_ylabel("Efficiency")
    ax.grid()
    return ax


def loop_on_trs(func, *args, ax=False, trs=np.linspace(0.1, 5, 4),TkEle=False, **kwargs):
    if not ax:
        _, ax = plt.subplots()
    for idx,tr in enumerate(np.tanh(trs)):

        if idx==len(trs)-1 and TkEle:
            func(*args, threshold=tr, ax=ax, TkEle=True, **kwargs)
        else:
            func(*args, threshold=tr, ax=ax, TkEle=False, **kwargs)
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.grid()
    return ax


def roc_pt(y_pred, y_test, pt_cuts, df_val, xlim=(0.000008, 0.1), log=True, ax=False):
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
    ax.legend(fontsize=18)
    if log:
        ax.set_xscale("log")

    if xlim:
        ax.set_xlim(xlim)
    hep.cms.text("Phase-2 Simulation")
    return ax




@nb.jit
def pt_y(ev_arr, pt_arr, y_arr):
    ev_idx = np.unique(ev_arr)
    nev = len(ev_idx)
    res_pt = np.zeros(nev)
    res_y = np.zeros(nev)
    for idx, ev in enumerate(ev_idx):
        mask = ev_arr == ev
        arg = np.argmax(pt_arr[mask])
        res_pt[idx] = pt_arr[mask][arg]
        res_y[idx] = y_arr[mask][arg]
    return res_pt, res_y


@nb.jit
def pt_cut(res_pt,res_y,pt_cuts,y_cuts):
    res=np.zeros((len(pt_cuts),len(y_cuts)))

    for pt_idx, pt in enumerate(pt_cuts):
        pt_mask = res_pt > pt
        for y_idx,y in enumerate(y_cuts):
            y_mask = res_y > y
            res[pt_idx,y_idx]=np.sum(np.bitwise_and(pt_mask,y_mask))/len(res_pt)
    return res




def rate_pt_plot(
    y_pred,
    df,
    pt_trs=np.linspace(0, 40, 15),
    y_trs=np.tanh(np.linspace(0., 6.5, 4)),
    log=False,
    ax=False,
):
    if not ax:
        _, ax = plt.subplots()


    ev_arr = df["CryClu_evIdx"].to_numpy()
    pt_arr = df["CryClu_pt"].to_numpy()

    res_pt, res_y = pt_y(ev_arr, pt_arr, y_pred)
    mat = pt_cut(res_pt, res_y, pt_trs, y_trs) * 11245.6 * 2500 / 1e3

    # *11245.6 * 2500 / 1e3
    for idx, s in enumerate(y_trs):
        print(idx)
        if idx > 5:
            style = "--"
        else:
            style = "-"
        ax.plot(pt_trs, mat[:,idx], style, label=f"score > {np.arctanh(s):.2f}")


    ax.set_xlabel("$p_T $ cut [GeV]")
    ax.set_ylabel("Trigger Rate [kHz]")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.grid()
    if log:
        ax.set_yscale("log")
    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    return ax



def get_matching_curve(bins, obj=None):
    assert obj=="CryClu" or obj=="TkEle"


    events = NanoEventsFactory.from_root(
        {root_signal: "Events"},
        schemaclass=NanoAODSchema,
    ).events()
    GenEle = events.GenEl



    GenEle = GenEle[np.abs(GenEle.eta) < 1.48]
    inAcceptanceMask = ak.num(GenEle) > 0
    GenEle = GenEle[inAcceptanceMask]
    GenEle["phi"] = GenEle.calophi
    GenEle["eta"] = GenEle.caloeta

    GenEle = GenEle.compute()

    if obj=="CryClu":


        CryClu = events.CaloCryCluGCT[inAcceptanceMask]
        Tk = events.DecTkBarrel[inAcceptanceMask]
        Tk["phi"] = Tk.caloPhi
        Tk["eta"] = Tk.caloEta
        Tk = Tk.compute()
        CryClu = CryClu.compute()
        CryClu["GenIdx"], GenEle["CryCluIdx"] = label_builder(
            ak.ArrayBuilder(),
            ak.ArrayBuilder(),
            CryClu,
            GenEle,
            dRcut=0.1
        )
        Tk["CryCluIdx"], CryClu["TkIdx"] = label_builder(
            ak.ArrayBuilder(), ak.ArrayBuilder(), Tk, CryClu, dRcut=0.1
        )
        Tk["GenIdx"], GenEle["TkIdx"] = label_builder(
            ak.ArrayBuilder(), ak.ArrayBuilder(), Tk, GenEle, dRcut=0.1
        )

        #tk_gen = GenEle[Tk.GenIdx]
        #cc_gen = GenEle[CryClu.GenIdx]
        tk_cc_gen = GenEle[Tk[CryClu[GenEle.CryCluIdx].TkIdx].GenIdx]
        hist, _ = np.histogram(ak.flatten(tk_cc_gen).pt, bins=bins)

    elif obj=="TkEle":
        TkEle = events.TkEleL2[inAcceptanceMask]
        TkEle = TkEle.compute()
        TkEle["GenIdx"], GenEle["TkEleIdx"] = label_builder(
            ak.ArrayBuilder(), ak.ArrayBuilder(), TkEle, GenEle, dRcut=0.1
        )
        tkele_gen = GenEle[TkEle.GenIdx]
        hist, _ = np.histogram(ak.flatten(tkele_gen).pt, bins=bins)

    genHist, _ = np.histogram(GenEle.pt, bins=bins, range=(0, 100))

    return hist/genHist



    # %%



def matching_plot(bins):
    _, ax = plt.subplots()
    centers = (bins[:-1] + bins[1:]) / 2
    ax.step(centers, get_matching_curve(bins, obj="CryClu"), where='mid', label="CryClu")
    ax.step(centers, get_matching_curve(bins, obj="TkEle"), where='mid', label="TkEle")
    ax.set_xlabel("genPt [GeV]")
    ax.set_ylabel("Matching Efficiency")
    ax.grid()
    ax.legend()
    return ax



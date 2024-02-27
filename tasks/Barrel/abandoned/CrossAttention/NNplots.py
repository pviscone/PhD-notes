import os
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
import corner
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mplhep as hep

hep.style.use("CMS")
savefig = False
img_format = "pdf"

def save(func):
    def wrapper(*args, **kwargs):
        fig, ax = func(*args, **kwargs)
        if savefig:
            fig.savefig(os.path.join(savefig, f"{func.__name__}.{img_format}"))
        return fig, ax

    return wrapper


def cms_label(ax,grid=True):
    hep.cms.text("Phase2 Simulation", ax=ax)
    hep.cms.lumitext("PU200 (14 TeV)", ax=ax)
    ax.legend()
    if grid:
        ax.grid()


#!Plot trigger rate
@save
def trigger_rate(y_pred, y_test):
    fig, ax = plt.subplots()
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel())
    # plot the eddiciecy vs trigger rate for phase 2
    ax.plot(fpr * 11245.6 * 2500 / 1e3, tpr)
    ax.set_xlabel("Trigger Rate [kHz]")
    ax.set_ylabel("Electron efficiency")
    ax.set_xlim(0, 40)
    cms_label(ax)
    return fig, ax


#! Plot confusion matrix
@save
def conf_matrix(y_pred, y_test):
    fig, ax = plt.subplots()
    conf = confusion_matrix(y_test.ravel(), y_pred.ravel() > 0.5, normalize="true")
    print(conf)
    im=ax.imshow(conf, cmap="Blues", interpolation="nearest")
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1], ["Background", "Signal"])
    ax.set_yticks([0, 1], ["Background", "Signal"])
    cms_label(ax,grid=False)
    return fig, ax


#!! Plot efficiency vs genPt
# genpt is the genPt of the gen that match the CryClu, 0 if there is no match
@save
def genPt_eff(genPt, y_pred, y_test, bins=50, histrange=(0, 100)):
    fig, ax = plt.subplots(2, 1)

    ptSig = genPt[y_test == 1]
    ptSigCorr = ptSig[y_pred[y_test == 1, 0] > 0.5]
    ptSigNotCorr = ptSig[y_pred[y_test == 1, 0] < 0.5]

    ax[0].hist(
        [ptSigCorr, ptSigNotCorr],
        bins=bins,
        range=histrange,
        label=["Correct", "Incorrect"],
        stacked=True,
    )
    ax[0].grid()
    # ax[0].set_xlabel('genPt [GeV]')
    ax[0].legend()
    cms_label(ax[0])

    hSig = np.histogram(ptSig, bins=bins, range=histrange)
    hSigCorr = np.histogram(ptSigCorr, bins=bins, range=histrange)
    centers = (hSig[1][1:] + hSig[1][:-1]) / 2
    eff = np.nan_to_num(hSigCorr[0] / hSig[0], 0)

    ax[1].step(centers, eff, where="mid", marker="v", label="Electron")
    ax[1].set_ylabel("Efficiency")
    ax[1].set_xlabel("genPt [GeV]")
    ax[1].grid()
    ax[1].legend()
    return fig, ax


#!Plot eta-phi efficiency
@save
def eta_phi_eff(
    eta, phi, y_pred, y_test, bins=(15, 10), histrange=((-3.14, 3.14), (-1.5, 1.5))
):
    eta = eta[y_test == 1]
    phi = phi[y_test == 1]

    fig, ax = plt.subplots()
    h2d = np.histogram2d(
        phi.ravel(),
        eta.ravel(),
        bins=bins,
        range=histrange,
    )

    etaCorr = eta[y_pred[y_test == 1, 0] > 0.5]
    phiCorr = phi[y_pred[y_test == 1, 0] > 0.5]

    h2dCorr = np.histogram2d(
        phiCorr.ravel(), etaCorr.ravel(), bins=bins, range=histrange
    )
    etaPhiEff = np.nan_to_num(h2dCorr[0] / h2d[0], 0)

    im = ax.imshow(
        etaPhiEff, extent=(-1.5, 1.5, -3.14, 3.14), origin="lower", cmap="viridis"
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical", label="Efficiency")
    ax.set_ylabel("$\phi$")
    ax.set_xlabel("$\eta$")
    cms_label(ax)
    return fig, ax


@save
def corner_TPvsFN(dataset, y_pred, y_test, varList):
    shape = dataset.shape
    dataset = np.reshape(dataset, (shape[0] * shape[1], shape[2]))

    dataset = dataset[y_test.ravel() == 1]

    datasetTP = dataset[y_pred[y_test == 1, 0] > 0.5]
    datasetFN = dataset[y_pred[y_test == 1, 0] < 0.5]

    fig = corner.corner(
        datasetTP,
        labels=varList,
        label_kwargs={"fontsize": 14},
        levels=(0.5, 0.9, 0.99),
        color="tab:blue",
    )

    corner.corner(
        datasetFN, labels=varList, fig=fig, levels=(0.5, 0.9, 0.99), color="tab:orange"
    )

    blue_line = mlines.Line2D([], [], color="tab:blue", label="Signal TP")
    red_line = mlines.Line2D([], [], color="tab:orange", label="Signal misclassified")
    plt.legend(
        handles=[blue_line, red_line],
        loc="upper right",
        frameon=False,
        bbox_to_anchor=(1, 5),
        fontsize=25,
    )

    return fig, plt.gca()

import numpy as np
import awkward as ak
import numba as nb


def flat(akarr):
    return ak.to_numpy(ak.drop_none(ak.flatten(akarr)))


@nb.jit(nopython=True)
def delta_phi(phi1, phi2):
    return (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi


@nb.jit(nopython=True)
def delta_r(phi1, phi2, eta1, eta2):
    return np.sqrt((eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2)


def snapshot_wrapper(func):
    def wrapper(*args, **kwargs):
        res12, res21 = func(*args, **kwargs)
        return res12.snapshot(), res21.snapshot()

    return wrapper


@snapshot_wrapper
@nb.jit(nopython=True)
def label_builder(builder12, builder21, Obj, ObjToLoopOn, dRcut):
    for event_idx in range(len(Obj)):
        builder12.begin_list()
        builder21.begin_list()
        obj = Obj[event_idx]
        res12 = np.ones(len(obj), dtype=nb.int32) * -1
        res21 = np.ones(len(ObjToLoopOn[event_idx]), dtype=nb.int32) * -1
        for idx, ele in enumerate(ObjToLoopOn[event_idx]):
            dr = delta_r(
                np.array(ele.phi),
                np.array(obj.phi),
                np.array(ele.eta),
                np.array(obj.eta),
            )
            # For each objToLoopOn, match the objwith dR<0.2 and the hightest pt
            if np.sum(dr < dRcut) >= 1:
                pt_arr = np.array(obj.pt)
                pt_arr[dr >= dRcut] = -1
                if np.max(pt_arr) > res12[np.argmax(pt_arr)]:
                    res12[np.argmax(pt_arr)] = idx
                res21[idx] = np.argmax(pt_arr)
        for elem in res12:
            if elem == -1:
                builder12.append(None)
            else:
                builder12.append(elem)
        for elem in res21:
            if elem == -1:
                builder21.append(None)
            else:
                builder21.append(elem)
        builder12.end_list()
        builder21.end_list()
    return builder12, builder21


def snap_wrapper(func):
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        return res.snapshot()

    return wrapper


@snap_wrapper
@nb.jit(nopython=True)
def evIdx(builder,obj):
    for ev in range(len(obj)):
        builder.begin_list()
        res=np.ones(len(obj[ev]),dtype=nb.int32)*ev
        for elem in res:
            builder.append(elem)
        builder.end_list()
    return builder
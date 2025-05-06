from .diff_compute_func import ComputeFunc
from params import Params


def compute(x, p : Params):
    a = p.get_a
    b = p.get_b
    w = p.get_w
    compute_func = ComputeFunc(x)
    out_y = compute_func(a, b, w)
    return out_y
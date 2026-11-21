"""ctypes binding for libsmshift.so (see include/smshift.h).

The library is looked up in $SMSHIFT_LIB, then ../build/libsmshift.so relative to
this file, then the system library path.

Launch interception is thread-local in the library: arm() and the kernel
launch it applies to must happen on the same Python thread.
"""
import ctypes
import ctypes.util
import os


def _load():
    candidates = [os.environ.get("SMSHIFT_LIB"),
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "libsmshift.so"),
                  ctypes.util.find_library("smshift")]
    for c in candidates:
        if c and (os.path.exists(c) or not os.path.dirname(c)):
            return ctypes.CDLL(c)
    raise OSError("libsmshift.so not found; build it with `make` or set SMSHIFT_LIB")


_lib = _load()

ERRORS = {-1: "CUDA call failed", -2: "unsupported GPU or driver hook unavailable",
          -3: "QMD version is not V04_00", -4: "invalid argument", -5: "no GPU VA below 1 TB"}


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what}: {ERRORS.get(rc, rc)}")


class SpawnArgs(ctypes.Structure):
    _fields_ = [
        ("tpcs", ctypes.POINTER(ctypes.c_int)),
        ("n", ctypes.c_int),
        ("resume", ctypes.c_uint),
        ("d_maxb", ctypes.c_void_p),
        ("margin", ctypes.c_uint),
        ("allowed_bits", ctypes.POINTER(ctypes.c_uint)),
        ("d_allowed", ctypes.c_void_p),
        ("d_counter", ctypes.c_void_p),
        ("d_at", ctypes.c_void_p),
        ("silent", ctypes.c_bool),
        ("grid_width", ctypes.c_uint),
    ]


class Kernel(ctypes.Structure):
    _fields_ = [("qmd_va", ctypes.c_uint64), ("tmpl", ctypes.c_int)]


_lib.smshift_init.argtypes = [ctypes.c_bool]
_lib.smshift_arm.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_lib.smshift_last_va.restype = ctypes.c_uint64
_lib.smshift_spawn.argtypes = [ctypes.c_void_p, ctypes.POINTER(Kernel), ctypes.POINTER(SpawnArgs)]
_lib.smshift_set_allowed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
_lib.smshift_fence.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
_lib.smshift_probe_sms.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
_lib.smshift_tpc_sms.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
_lib.smshift_valid_tpcs.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]


def _ints(xs):
    return (ctypes.c_int * max(1, len(xs)))(*xs)


def _bits(xs):
    return (ctypes.c_uint * 8)(*xs)


def init(sibling=True):
    _check(_lib.smshift_init(sibling), "smshift_init")


def arm(tpcs):
    _lib.smshift_arm(_ints(tpcs), len(tpcs))


def arm_all():
    _lib.smshift_arm_all()


def last_va():
    return _lib.smshift_last_va()


def kernel_handle():
    """Handle for the QMD of this thread's most recent launch."""
    va = last_va()
    if not va:
        raise RuntimeError("no QMD captured for the last launch")
    return Kernel(va, -1)


def probe_sms(tpcs):
    out = (ctypes.c_uint * 8)()
    _lib.smshift_probe_sms(_ints(tpcs), len(tpcs), out)
    return list(out)


def spawn(side_stream, k, tpcs, resume=0, allowed_bits=None, d_allowed=0, d_maxb=0, margin=0, silent=True,
          grid_width=0):
    keep = (_ints(tpcs), _bits(allowed_bits) if allowed_bits is not None else None)
    a = SpawnArgs(keep[0], len(tpcs), resume, d_maxb or None, margin, keep[1], d_allowed or None, None, None, silent,
                  grid_width)
    _check(_lib.smshift_spawn(side_stream, ctypes.byref(k), ctypes.byref(a)), "smshift_spawn")


def set_allowed(side_stream, d_allowed, bits):
    _check(_lib.smshift_set_allowed(side_stream, d_allowed, _bits(bits)), "smshift_set_allowed")


def fence(stream, d_value, target):
    _check(_lib.smshift_fence(stream, d_value, target), "smshift_fence")


def tpc_sms(bit):
    """SM bitmap (8 words) behind one TPC_DISABLE_MASK bit; all zero if none."""
    out = (ctypes.c_uint * 8)()
    _check(_lib.smshift_tpc_sms(bit, out), "smshift_tpc_sms")
    return list(out)


_valid_cache = None


def valid_tpcs():
    """Mask bits that have SMs, in increasing order (the layout has gaps)."""
    global _valid_cache
    if _valid_cache is None:
        buf = (ctypes.c_int * 128)()
        n = _lib.smshift_valid_tpcs(buf, 128)
        if n < 0:
            _check(n, "smshift_valid_tpcs")
        _valid_cache = list(buf[:n])
    return _valid_cache


def popcount(bits):
    return sum(bin(b).count("1") for b in bits)

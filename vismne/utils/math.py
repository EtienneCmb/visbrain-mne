"""Math functions."""

import numpy as np


def normalize(x, tomin=0., tomax=1.):
    """Normalize the array x between tomin and tomax.

    Parameters
    ----------
    x : array_like
        The array to normalize
    tomin : int/float | 0.
        Minimum of returned array

    tomax : int/float | 1.
        Maximum of returned array

    Returns
    -------
    xn : array_like
        The normalized array
    """
    if x.size:
        x = np.float32(x)
        xm, xh = np.float32(x.min()), np.float32(x.max())
        if xm != xh:
            coef = (tomax - tomin) / (xh - xm)
            np.subtract(x, xh, out=x)
            np.multiply(x, coef, out=x)
            np.add(x, tomax, out=x)
            return x
            # return tomax - (((tomax - tomin) * (xh - x)) / (xh-xm))
        else:
            np.multiply(x, tomax, out=x)
            np.divide(x, xh, out=x)
            return x
    else:
        return x


def vispy_array(data, dtype=np.float32):
    """Check and convert array to be compatible with buffers.

    Parameters
    ----------
    data : array_like
        Array of data.
    dtype : type | np.float32
        Futur type of the array.

    Returns
    -------
    data : array_like
        Contiguous array of type dtype.
    """
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data, dtype=dtype)
    if data.dtype != dtype:
        data = data.astype(dtype, copy=False)
    return data

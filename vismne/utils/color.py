"""Color functions."""
from warnings import warn
import logging

import numpy as np

from matplotlib import cm
import matplotlib.colors as mplcol

from .math import vispy_array


logger = logging.getLogger('visbrain-mne')


class Colormap(object):
    """Main colormap class.

    Parameters
    ----------
    cmap : string | inferno
        Matplotlib colormap
    clim : tuple/list | None
        Limit of the colormap. The clim parameter must be a tuple / list
        of two float number each one describing respectively the (min, max)
        of the colormap. Every values under clim[0] or over clim[1] will
        peaked.
    alpha : float | 1.0
        The opacity to use. The alpha parameter must be between 0 and 1.
    vmin : float | None
        Threshold from which every color will have the color defined using
        the under parameter bellow.
    under : tuple/string | 'dimgray'
        Matplotlib color for values under vmin.
    vmax : float | None
        Threshold from which every color will have the color defined using
        the over parameter bellow.
    over : tuple/string | 'darkred'
        Matplotlib color for values over vmax.
    translucent : tuple | None
        Set a specific range translucent. With f_1 and f_2 two floats, if
        translucent is :

            * (f_1, f_2) : values between f_1 and f_2 are set to translucent
            * (None, f_2) x <= f_2 are set to translucent
            * (f_1, None) f_1 <= x are set to translucent
    lut_len : int | 1024
        Number of levels for the colormap.
    interpolation : {None, 'linear', 'cubic'}
        Interpolation type. Default is None.

    Attributes
    ----------
    data : array_like
        Color data of shape (n_data, 4)
    shape : tuple
        Shape of the data.
    r : array_like
        Red levels.
    g : array_like
        Green levels.
    b : array_like
        Blue levels.
    rgb : array_like
        RGB levels.
    alpha : array_like
        Transparency level.
    glsl : vispy.colors.Colormap
        GL colormap version.
    """

    def __init__(self, cmap='viridis', clim=None, vmin=None, under=None,
                 vmax=None, over=None, translucent=None, alpha=1.,
                 lut_len=1024, interpolation=None):
        """Init."""
        # Keep color parameters into a dict :
        self._kw = dict(cmap=cmap, clim=clim, vmin=vmin, vmax=vmax,
                        under=under, over=over, translucent=translucent,
                        alpha=alpha)
        # Color conversion :
        if isinstance(cmap, np.ndarray):
            assert (cmap.ndim == 2) and (cmap.shape[-1] in (3, 4))
            # cmap = single color :
            if (cmap.shape[0] == 1) and isinstance(interpolation, str):
                logger.debug("Colormap : unique color repeated.")
                data = np.tile(cmap, (lut_len, 1))
            elif (cmap.shape[0] == lut_len) or (interpolation is None):
                logger.debug("Colormap : Unique repeated.")
                data = cmap
            else:
                from scipy.interpolate import interp2d
                n_ = cmap.shape[1]
                x, y = np.linspace(0, 1, n_), np.linspace(0, 1, cmap.shape[0])
                f = interp2d(x, y, cmap, kind=interpolation)
                # Interpolate colormap :
                data = f(x, np.linspace(0, 1, lut_len))
        elif isinstance(cmap, str):
            data = array_to_color(np.linspace(0., 1., lut_len), **self._kw)
        # Alpha correction :
        if data.shape[-1] == 3:
            data = np.c_[data, np.full((data.shape[0],), alpha)]
        # NumPy float32 conversion :
        self._data = vispy_array(data)

    def to_rgba(self, data):
        """Turn a data vector into colors using colormap properties.

        Parameters
        ----------
        data : array_like
            Vector of data of shape (n_data,).

        Returns
        -------
        color : array_like
            Array of colors of shape (n_data, 4)
        """
        if isinstance(self._kw['cmap'], np.ndarray):
            return self._data
        else:
            return array_to_color(data, **self._kw)

    def __len__(self):
        """Get the number of colors in the colormap."""
        return self._data.shape[0]

    def __getitem__(self, name):
        """Get a color item."""
        return self._kw[name]

    @property
    def data(self):
        """Get colormap data."""
        return self._data

    @property
    def shape(self):
        """Get the shape of the data."""
        return self._data.shape

    # @property
    # def glsl(self):
    #     """Get a glsl version of the colormap."""
    #     return cmap_to_glsl(lut_len=len(self), **self._kw)

    @property
    def r(self):
        """Get red levels."""
        return self._data[:, 0]

    @property
    def g(self):
        """Get green levels."""
        return self._data[:, 1]

    @property
    def b(self):
        """Get blue levels."""
        return self._data[:, 2]

    @property
    def rgb(self):
        """Get rgb levels."""
        return self._data[:, 0:3]

    @property
    def alpha(self):
        """Get transparency level."""
        return self._data[:, -1]


def color2vb(color=None, default=(1., 1., 1.), length=1, alpha=1.0,
             faces_index=False):
    """Turn into a RGBA compatible color format.

    This function can tranform a tuple of RGB, a matplotlib color or an
    hexadecimal color into an array of RGBA colors.

    Parameters
    ----------
    color : None/tuple/string | None
        The color to use. Can either be None, or a tuple (R, G, B),
        a matplotlib color or an hexadecimal color '#...'.
    default : tuple | (1,1,1)
        The default color to use instead.
    length : int | 1
        The length of the output array.
    alpha : float | 1
        The opacity (Last digit of the RGBA tuple).
    faces_index : bool | False
        Specify if the returned color have to be compatible with faces index
        (e.g a (n_color, 3, 4) array).

    Return
    ------
    vcolor : array_like
        Array of RGBA colors of shape (length, 4).
    """
    # Default or static color :
    if (color is None) or isinstance(color, (str, tuple, list, np.ndarray)):
        if color is None:  # Default
            coltuple = default
        elif isinstance(color, (tuple, list, np.ndarray)):  # Static
            color = np.squeeze(color).ravel()
            if len(color) == 4:
                alpha = color[-1]
                color = color[0:-1]
            coltuple = color
        elif isinstance(color, str) and (color[0] is not '#'):  # Matplotlib
            # Check if the name is in the Matplotlib database :
            if color in mplcol.cnames.keys():
                coltuple = mplcol.hex2color(mplcol.cnames[color])
            else:
                warn("The color name " + color + " is not in the matplotlib "
                     "database. Default color will be used instead.")
                coltuple = default
        elif isinstance(color, str) and (color[0] is '#'):  # Hexadecimal
            try:
                coltuple = mplcol.hex2color(color)
            except:
                warn("The hexadecimal color " + color + " is not valid. "
                     "Default color will be used instead.")
                coltuple = default
        # Set the color :
        vcolor = np.concatenate((np.array([list(coltuple)] * length),
                                 alpha * np.ones((length, 1),
                                                 dtype=np.float32)), axis=1)

        # Faces index :
        if faces_index:
            vcolor = np.tile(vcolor[:, np.newaxis, :], (1, 3, 1))

        return vcolor.astype(np.float32)
    else:
        raise ValueError(str(type(color)) + " is not a recognized type of "
                         "color. Use None, tuple or string")


def array_to_color(x, cmap='inferno', clim=None, alpha=1.0, vmin=None,
                   vmax=None, under='dimgray', over='darkred',
                   translucent=None, faces_render=False):
    """Transform an array of data into colormap (array of RGBA).

    Parameters
    ----------
    x: array
        Array of data
    cmap : string | inferno
        Matplotlib colormap
    clim : tuple/list | None
        Limit of the colormap. The clim parameter must be a tuple / list
        of two float number each one describing respectively the (min, max)
        of the colormap. Every values under clim[0] or over clim[1] will
        peaked.
    alpha : float | 1.0
        The opacity to use. The alpha parameter must be between 0 and 1.
    vmin : float | None
        Threshold from which every color will have the color defined using
        the under parameter bellow.
    under : tuple/string | 'dimgray'
        Matplotlib color for values under vmin.
    vmax : float | None
        Threshold from which every color will have the color defined using
        the over parameter bellow.
    over : tuple/string | 'darkred'
        Matplotlib color for values over vmax.
    translucent : tuple | None
        Set a specific range translucent. With f_1 and f_2 two floats, if
        translucent is :

            * (f_1, f_2) : values between f_1 and f_2 are set to translucent
            * (None, f_2) x <= f_2 are set to translucent
            * (f_1, None) f_1 <= x are set to translucent
    faces_render : boll | False
        Precise if the render should be applied to faces

    Returns
    -------
    color : array_like
        Array of RGBA colors
    """
    # ================== Check input argument types ==================
    # Force data to be an array :
    x = np.asarray(x)

    # Check clim :
    clim = (None, None) if clim is None else list(clim)
    assert len(clim) == 2

    # ---------------------------
    # Check alpha :
    if (alpha < 0) or (alpha > 1):
        warn("The alpha parameter must be >= 0 and <= 1.")

    # ================== Define colormap ==================
    sc = cm.ScalarMappable(cmap=cmap)

    # Fix limits :
    norm = mplcol.Normalize(vmin=clim[0], vmax=clim[1])
    sc.set_norm(norm)

    # ================== Apply colormap ==================
    # Apply colormap to x :
    x_cmap = np.array(sc.to_rgba(x, alpha=alpha))

    # ================== Colormap (under, over) ==================
    if (vmin is not None) and (under is not None):
        under = color2vb(under)
        x_cmap[x < vmin, :] = under
    if (vmax is not None) and (over is not None):
        over = color2vb(over)
        x_cmap[x > vmax, :] = over

    # ================== Transparency ==================
    x_cmap = _transclucent_cmap(x, x_cmap, translucent)

    return x_cmap.astype(np.float32)


def _transclucent_cmap(x, x_cmap, translucent, smooth=None):
    """Sub function to define transparency."""
    if translucent is not None:
        is_num = [isinstance(k, (int, float)) for k in translucent]
        assert len(translucent) == 2 and any(is_num)
        if all(is_num):                # (f_1, f_2)
            trans_x = np.logical_and(translucent[0] <= x, x <= translucent[1])
        elif is_num == [True, False]:  # (f_1, None)
            trans_x = translucent[0] <= x
        elif is_num == [False, True]:  # (None, f_2)
            trans_x = x <= translucent[1]
        x_cmap[..., -1] = np.invert(trans_x)
        if isinstance(smooth, int):
            alphas = x_cmap[:, -1]
            alphas = np.convolve(alphas, np.hanning(smooth), 'valid')
            alphas /= max(alphas.max(), 1.)
            x_cmap[smooth - 1::, -1] = alphas
    return x_cmap

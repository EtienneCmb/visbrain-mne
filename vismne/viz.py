"""Need doc."""
from .io import _get_subjects_dir
from .utils import Surface, _check_units
from .visuals import BrainMesh


lh_viewdict = {'lateral': {'v': (180., 90.), 'r': 90.},
               'medial': {'v': (0., 90.), 'r': -90.},
               'rostral': {'v': (90., 90.), 'r': -180.},
               'caudal': {'v': (270., 90.), 'r': 0.},
               'dorsal': {'v': (180., 0.), 'r': 90.},
               'ventral': {'v': (180., 180.), 'r': 90.},
               'frontal': {'v': (120., 80.), 'r': 106.739},
               'parietal': {'v': (-120., 60.), 'r': 49.106}}
rh_viewdict = {'lateral': {'v': (180., -90.), 'r': -90.},
               'medial': {'v': (0., -90.), 'r': 90.},
               'rostral': {'v': (-90., -90.), 'r': 180.},
               'caudal': {'v': (90., -90.), 'r': 0.},
               'dorsal': {'v': (180., 0.), 'r': 90.},
               'ventral': {'v': (180., 180.), 'r': 90.},
               'frontal': {'v': (60., 80.), 'r': -106.739},
               'parietal': {'v': (-60., 60.), 'r': -49.106}}
viewdicts = dict(lh=lh_viewdict, rh=rh_viewdict)


class Brain(object):
    """docstring for Brain."""

    def __init__(self, subject_id, hemi, surf, title=None, views=['lat'],
                 offset=True):
        self._units = _check_units(units)
        col_dict = dict(lh=1, rh=1, both=1, split=2)
        n_col = col_dict[hemi]
        if hemi not in col_dict.keys():
            raise ValueError('hemi must be one of [%s], not %s'
                             % (', '.join(col_dict.keys()), hemi))
        # Get the subjects directory from parameter or env. var
        subjects_dir = _get_subjects_dir(subjects_dir=subjects_dir)

        self._hemi = hemi
        if title is None:
            title = subject_id
        self.subject_id = subject_id

        if not isinstance(views, list):
            views = [views]
        n_row = len(views)

        # load geometry for one or both hemispheres as necessary
        offset = None if (not offset or hemi != 'both') else 0.0
        self.geo = dict()
        if hemi in ['split', 'both']:
            geo_hemis = ['lh', 'rh']
        elif hemi == 'lh':
            geo_hemis = ['lh']
        elif hemi == 'rh':
            geo_hemis = ['rh']
        else:
            raise ValueError('bad hemi value')
        # geo_kwargs, geo_reverse, geo_curv = self._get_geo_params(cortex, alpha)
        for h in geo_hemis:
            # Initialize a Surface object as the geometry
            geo = Surface(subject_id, h, surf, subjects_dir, offset,
                          units=self._units)
            # Load in the geometry and (maybe) curvature
            geo.load_geometry()
            if geo_curv:
                geo.load_curvature()
            self.geo[h] = geo

    ###########################################################################
    # ADDING DATA PLOTS
    def add_overlay(self):
        """Doc."""
        raise NotImplementedError

    def add_data(self):
        """Doc."""
        raise NotImplementedError

    def add_annotation(self):
        """Doc."""
        raise NotImplementedError

    def add_label(self):
        """Doc."""
        raise NotImplementedError

    def remove_data(self):
        """Doc."""
        raise NotImplementedError

    def remove_labels(self):
        """Doc."""
        raise NotImplementedError

    def add_morphometry(self):
        """Doc."""
        raise NotImplementedError

    def add_foci(self):
        """Doc."""
        raise NotImplementedError

    def add_contour_overlay(self):
        """Doc."""
        raise NotImplementedError

    def add_text(self):
        """Doc."""
        raise NotImplementedError

    def update_text(self):
        """Doc."""
        raise NotImplementedError

    def reset_view(self):
        """Doc."""
        raise NotImplementedError

    def show_view(self):
        """Doc."""
        raise NotImplementedError

    def set_distance(self):
        """Doc."""
        raise NotImplementedError

    def set_surf(self):
        """Doc."""
        raise NotImplementedError

    def scale_data_colormap(self):
        """Doc."""
        raise NotImplementedError

    def set_data_time_index(self):
        """Doc."""
        raise NotImplementedError

    def data_time_index(self):
        """Doc."""
        raise NotImplementedError

    def set_data_smoothing_steps(self):
        """Doc."""
        raise NotImplementedError

    def index_for_time(self):
        """Doc."""
        raise NotImplementedError

    def set_time(self):
        """Doc."""
        raise NotImplementedError

    def show_colorbar(self):
        """Doc."""
        raise NotImplementedError

    def hide_colorbar(self):
        """Doc."""
        raise NotImplementedError

    def close(self):
        """Doc."""
        raise NotImplementedError

    ###########################################################################
    # SAVING
    def save_single_image(self):
        """Doc."""
        raise NotImplementedError

    def save_image(self):
        """Doc."""
        raise NotImplementedError

    def screenshot(self):
        """Doc."""
        raise NotImplementedError

    def screenshot_single(self):
        """Doc."""
        raise NotImplementedError

    def save_imageset(self):
        """Doc."""
        raise NotImplementedError

    def save_image_sequence(self):
        """Doc."""
        raise NotImplementedError

    def save_montage(self):
        """Doc."""
        raise NotImplementedError

    def save_movie(self):
        """Doc."""
        raise NotImplementedError

    def animate(self):
        """Doc."""
        raise NotImplementedError

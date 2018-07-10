"""Need doc."""


class Brain(object):
    """docstring for Brain."""

    def __init__(self):

        pass

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

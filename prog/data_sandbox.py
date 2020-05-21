
from generic_sublayouts import SubLayout

from bokeh.models import PointDrawTool, Div, RadioButtonGroup, TextInput, Toggle, \
    Button, Slider, RangeSlider, ColumnDataSource, FreehandDrawTool
from bokeh import events
from bokeh.layouts import row, column
from bokeh.plotting import figure


from constants import CLUSTER_SIZE_DEF, CLUSTER_DEV_DEF, CLUSTER_SIZE_MAX, \
    SAVED_DATASET_FILE_NAME, EMPTY_VALUE_COLOR, LASSO_SLIDER_END, LASSO_SLIDER_START, LASSO_SLIDER_STARTING_VAL, \
    LASSO_SLIDER_STEP, CLUSTER_RANGE_X, CLUSTER_RANGE_Y, CLUSTER_RANGE_STEP, FREEHAND_DENSITY_START, \
    FREEHAND_DENSITY_END, FREEHAND_DENSITY_STEP, FREEHAND_DENSITY_STARTING_VAL, FREEHAND_DEVIATION_END, \
    FREEHAND_DEVIATION_START, FREEHAND_DEVIATION_STARTING_VAL, FREEHAND_DEVIATION_STEP, \
    CLUSTER_DEV_MAX, ALPHA_DEF, BETA_DEF, BETA_MODE, UNIFORM_MODE


STANDARD_MODE = "Standard"
LASSO_APPEND = "Lasso append"
FREEHAND_APPEND = "Freehand"

# classification data sandbox modes
GENERATE_NEW_CLUSTERS = "New clusters"

# local constants
BETA_CONST = 1


class DataSandbox(SubLayout):
    def __init__(self, name, source_data):
        self.source_data = source_data
        SubLayout.__init__(self, name)

        self.source_data.plot_source.on_change('data', self._plot_source_change)  # DataSandbox can update statistics

        self._freehand_source = ColumnDataSource(data=dict(x=[], y=[]))
        freehand_renderer = self._fig.multi_line('x', 'y',
                                                 source=self._freehand_source, color='black')
        self._freehand_tool = FreehandDrawTool(renderers=[freehand_renderer])
        self._fig.add_tools(self._freehand_tool)

        """Set up Lasso and FreeHand triggers"""
        self._freehand_source.on_change('data', self._freehand_add)
        self._fig.on_event(events.SelectionGeometry, self._lasso_add)

    """Methods for sublayout initialisation"""

    def _layout_init(self):
        fig_layout = self._init_figure()
        button_layout = self._init_button_layout()
        return row(fig_layout, button_layout)
    
    def _init_button_layout(self):
        basic_buttons = self._init_basic_buttons()
        self._create_distribution_options()  # creates attribute _distribution_options and options for dist. settings

        self._generation_modes = {
            STANDARD_MODE: self._init_standard_option(),
            LASSO_APPEND: self._init_lasso_option(),
            FREEHAND_APPEND: self._init_freehand_option()
        }
        self._add_special_modes()

        mode_button_labels = list(self._generation_modes.keys())
        mode_button_width = 20  # 60 * (len(mode_button_labels) + 1)
        self._points_generation_mode_button = RadioButtonGroup(
            labels=mode_button_labels, active=0, width_policy="fixed")
        self._points_generation_mode_button.on_change('active', self._points_generation_mode_trigger)
        mode_text = Div(text="Data sandbox mode: ", style={'font-size': '150%'})

        active_mode = self._generation_modes[STANDARD_MODE]

        self._am1, self._am2 = 1, 3  # active mode position
        return column(basic_buttons,
                      mode_text,
                      self._points_generation_mode_button,
                      active_mode,
                      )

    def _add_special_modes(self):
        pass

    def _init_basic_buttons(self):
        self._data_size_info = Div(text="Points in dataset: " + str(len(self.source_data.plot_source.data[self.source_data.x])))

        self._save_button = Button(label="Save dataset", button_type="primary")
        self._save_button.on_click(self._save_dataset)

        self._save_path = TextInput(value=SAVED_DATASET_FILE_NAME, title="Name of the file:")
        self._save_info = Div(text="", style={'font-size': '75%'}, width_policy="fixed")
        save_group = column(self._save_path, self._save_button, self._save_info)

        return column(self._data_size_info,
                      save_group)

    def _init_lasso_option(self):
        lasso_general_info = self._lasso_general_info()

        self._lasso_density_slider = Slider(start=LASSO_SLIDER_START, end=LASSO_SLIDER_END,
                                            step=LASSO_SLIDER_STEP, value=LASSO_SLIDER_STARTING_VAL,
                                            title="Size")

        distribution_options = self._distribution_options

        return column(lasso_general_info,
                      self._lasso_density_slider,
                      distribution_options)

    def _init_freehand_option(self):
        freehand_info = Div(text="Add points around a line you draw by selecting"
                                 " FreeHand Draw Tool in the toolbar")

        self._freehand_density = Slider(start=FREEHAND_DENSITY_START, end=FREEHAND_DENSITY_END,
                                        step=FREEHAND_DENSITY_STEP, value=FREEHAND_DENSITY_STARTING_VAL,
                                        title="Density")
        self._freehand_volatility = Slider(start=FREEHAND_DEVIATION_START, end=FREEHAND_DEVIATION_END,
                                           step=FREEHAND_DEVIATION_STEP, value=FREEHAND_DEVIATION_STARTING_VAL,
                                           title="Volatility (+-)")
        distribution_options = self._distribution_options

        return column(freehand_info,
                      self._freehand_density,
                      self._freehand_volatility,
                      distribution_options)

    def _create_distribution_options(self):
        """Create an attribute with distribution options uniform or beta and sliders for beta parameters.
        This method sets widgets into _distribution_options and DOES NOT RETURN ANYTHING."""
        mode_text = Div(text="Distribution:")
        self._distribution_mode_button = RadioButtonGroup(labels=[UNIFORM_MODE, BETA_MODE],
                                                          active=1)
        beta_text = Div(text="Beta distribution options:")
        self._beta_random_toggle = Toggle(label="Random alpha beta", active=False)
        self._distribution_alpha_slider = Slider(start=0.2, end=5, step=0.2, value=ALPHA_DEF, title="Alpha")
        self._distribution_beta_slider = Slider(start=0.2, end=5, step=0.2, value=BETA_DEF, title="Beta")
        self._distribution_alpha_slider.on_change('value', self._beta_plot_change)
        self._distribution_beta_slider.on_change('value', self._beta_plot_change)

        self._beta_plot = figure(x_axis_label=None, y_axis_label=None, toolbar_location=None,
                                 x_axis_location=None, y_axis_location=None,
                                 plot_width=250, plot_height=int(250*0.66),
                                 x_range=(0, 1), y_range=(0, 4))
        x_arr = [i/100 for i in range(1, 100)]
        self._beta_plot_source = ColumnDataSource(data=dict(
            x=x_arr,
            y=[0] * len(x_arr))
        )
        self._beta_plot.line(x='x', y='y', source=self._beta_plot_source)
        self._beta_plot_change(None, None, None)

        self._distribution_options = column(mode_text,
                                            self._distribution_mode_button,
                                            beta_text,
                                            self._beta_random_toggle,
                                            self._distribution_alpha_slider,
                                            self._distribution_beta_slider,
                                            self._beta_plot)

    @staticmethod
    def _init_standard_option():
        return Div(
            text="You can move with points or delete the with Point Draw Tool or Lasso. "
                 "For adding points with Lasso click on \"append with Lasso\".")

    @staticmethod
    def _lasso_general_info():
        return Div(
            text="Add new points by selecting \"Lasso Select\" in the figure toolbar. "
                 "Then draw a polygon in the figure. "
                 "If some points are not visible properly, click on \"Reset\" "
                 "in the figure toolbar for resetting the view.")

    """Methods for interactive calling (called by on_change or on_click triggers)"""

    def _points_generation_mode_trigger(self, attr, old, new):
        new_mode_str = self._points_generation_mode_button.labels[new]
        new_mode = self._generation_modes[new_mode_str]
        self.layout.children[self._am1].children[self._am2] = new_mode

    def _plot_source_change(self, attr, old, new):
        self._data_size_info.update(text="Points in dataset: " + str(len(new[self.source_data.x])))

    def _lasso_add(self, event):
        if event.final and self._is_lasso_append_active():

            keys = event.geometry['x'].keys()  # 'x' and 'y' have the same keys which is the number of vertex
            vertices = []
            for key in keys:
                vertices.append((event.geometry['x'][key], event.geometry['y'][key]))

            self._append_polygon(vertices)

    def _freehand_add(self, attr, old, new):
        self._append_freehand()

        self._freehand_source.remove_on_change('data', self._freehand_add)
        self._freehand_source.update(data=dict(x=[], y=[]))
        self._freehand_source.on_change('data', self._freehand_add)

    def _save_dataset(self):
        self._info("Saving dataset...")
        abs_path = self.source_data.save_source(self._save_path.value)
        self._save_info.update(text="Saved in: " + abs_path)
        self._info("Saved in " + str(abs_path))
        self._info("Saving DONE")

    def _beta_plot_change(self, attr, old, new):
        alpha = self._distribution_beta_slider.value
        beta = self._distribution_alpha_slider.value

        x_arr = [c/100 for c in range(1, 100)]
        y_arr = [BETA_CONST * pow(x, alpha - 1) * pow(1 - x, beta - 1) for x in x_arr]  # prob. dens. function

        self._beta_plot.y_range.end = min(10.0, max(y_arr) * 1.2)
        self._beta_plot_source.update(
            data=dict(
                x=x_arr,
                y=y_arr
            )
        )

    """Other methods"""

    def _append_polygon(self, vertices):
        pass

    def _append_freehand(self):
        pass

    def _get_distribution_params(self):
        mode = self._distribution_mode_button.labels[self._distribution_mode_button.active]
        beta_random = self._beta_random_toggle.active
        alpha = self._distribution_alpha_slider.value
        beta = self._distribution_beta_slider.value
        return mode, beta_random, alpha, beta

    def _get_lasso_density(self):
        return self._lasso_density_slider.value

    def _is_lasso_append_active(self):
        active_mode = self._points_generation_mode_button.labels[
            self._points_generation_mode_button.active]
        return active_mode == LASSO_APPEND


class RegressionDataSandbox(DataSandbox):
    def __init__(self, name, source_data):
        DataSandbox.__init__(self, name, source_data)

        move_circle = self._fig.circle(self.source_data.x, self.source_data.y, source=source_data.plot_source, size=7)
        point_draw_tool = PointDrawTool(renderers=[move_circle], empty_value=EMPTY_VALUE_COLOR, add=True)
        self._fig.add_tools(point_draw_tool)

    """Other methods"""

    def _append_polygon(self, vertices):
        density = self._get_lasso_density()
        distribution_params = self._get_distribution_params()
        self.source_data.append_polygon_data(vertices, density, distribution_params)

    def _append_freehand(self):
        [x_line] = self._freehand_source.data['x']
        [y_line] = self._freehand_source.data['y']
        size = self._freehand_density.value
        volatility = self._freehand_volatility.value
        distribution_params = self._get_distribution_params()
        
        self.source_data.append_freehand_data(x_line, y_line, size, volatility, distribution_params)


class ClassifierDataSandbox(DataSandbox):
    def __init__(self, name, source_data, class_select_button):
        DataSandbox.__init__(self, name, source_data)

        move_circle = self._fig.circle(x=self.source_data.x, y=self.source_data.y, color='color',
                                       source=source_data.plot_source, size=7)
        point_draw_tool = PointDrawTool(renderers=[move_circle], empty_value=EMPTY_VALUE_COLOR, add=True)
        self._fig.add_tools(point_draw_tool)

        self.__class_select_button = class_select_button

    def __del__(self):
        self.source_data.plot_source.remove_on_change('data', self._plot_source_change)  # removing trigger

    def update_slider_classes_count(self):
        """Update classes count on __cluster_count_slider."""
        self.__cluster_count_slider.update(end=len(self.source_data.uniq_values()))

    """Methods for sublayout initialisation"""

    def _add_special_modes(self):
        self._generation_modes[GENERATE_NEW_CLUSTERS] = self.__init_cluster_generating_options()

    def __init_cluster_generating_options(self):
        __generate_general_info = Div(
            text="Automaticly generate a new clusters. "
                 "You can choose whether the new cluster should replace data or replace them. "
                 "\"Clusters count\" option determines the number of cluster (each has it onw class), "
                 "\"size\" sets the size of clusters and \"±\" sets the volatility of size.")

        __generate_info = Div(text="New clusters: ", style={'font-size': '150%'})

        classes_count = len(self.source_data.uniq_values())
        self.__cluster_count_slider = Slider(start=1, end=classes_count, step=1, value=classes_count,
                                             title="Classes count (push 'add class' to add a new one)")
        self.__cluster_size_slider = Slider(start=1, end=CLUSTER_SIZE_MAX, step=1, value=CLUSTER_SIZE_DEF,
                                            title="Clusters size:")
        self.__cluster_deviation_slider = Slider(start=0, end=CLUSTER_DEV_MAX, step=1, value=CLUSTER_DEV_DEF,
                                                 title="Size deviation:")

        self.__new_clusters_mode_button = RadioButtonGroup(
            labels=["Replace", "Append"], width=200, active=0, sizing_mode="fixed")
        __new_clusters_mode_text = Div(text="Clusters adding mode: ")

        self.__cluster_x_range_slider = RangeSlider(start=CLUSTER_RANGE_X[0], end=CLUSTER_RANGE_X[1],
                                                    step=CLUSTER_RANGE_STEP,
                                                    value=CLUSTER_RANGE_X, title="x range")

        self.__cluster_y_range_slider = RangeSlider(start=CLUSTER_RANGE_Y[0], end=CLUSTER_RANGE_Y[1],
                                                    step=CLUSTER_RANGE_STEP,
                                                    value=CLUSTER_RANGE_Y, title="y_range")

        __generate_new_clusters_button = Button(label="Generate", button_type="primary")
        __generate_new_clusters_button.on_click(self.__generate_new_clusters)

        return column(__generate_general_info,
                      __generate_info,
                      row(__new_clusters_mode_text, self.__new_clusters_mode_button),
                      self.__cluster_count_slider,
                      self.__cluster_size_slider,
                      self.__cluster_deviation_slider,
                      self.__cluster_x_range_slider,
                      self.__cluster_y_range_slider,
                      __generate_new_clusters_button
                      )

    """Methods for interactive calling (called by on_change or on_click triggers)"""

    def __generate_new_clusters(self):
        self._info("Generating new dataset...")

        count, density, volatility, x_range, y_range = \
            self.__get_cluster_generating_params()
        replace = (0 == self.__new_clusters_mode_button.active)
        self.source_data.new_clusters(count, density, volatility, x_range, y_range, replace)

        self._info("Generating new dataset DONE")

    """Other methods"""

    @staticmethod
    def __get_int_set_error(text_input, lowest_val):
        """Returns POSITIVE INTEGER from text_input, if the value is different, returns "default" instead"""
        try:
            val = int(text_input.value)
        except ValueError:
            text_input.update(value="ERR")
            return None
        if val < lowest_val:
            text_input.update(value="ERR")
            return None
        return val

    def __get_cluster_generating_params(self):
        """Get parameters for generating clusters"""
        clusters_count = self.__cluster_count_slider.value
        clusters_size = self.__cluster_size_slider.value
        cluster_dev = self.__cluster_deviation_slider.value

        x_range = self.__cluster_x_range_slider.value
        y_range = self.__cluster_y_range_slider.value
        return clusters_count, clusters_size, cluster_dev, x_range, y_range

    def _append_polygon(self, vertices):
        distribution_params = self._get_distribution_params()
        density = self._get_lasso_density()
        cls = self.__get_actual_class()
        self.source_data.append_polygon_data(vertices, cls, density, distribution_params)

    def _append_freehand(self):
        [x_line] = self._freehand_source.data['x']
        [y_line] = self._freehand_source.data['y']
        size = self._freehand_density.value
        volatility = self._freehand_volatility.value
        cls = self.__get_actual_class()
        distribution_params = self._get_distribution_params()

        self.source_data.append_freehand_data(
            x_line, y_line, cls, size, volatility, distribution_params
        )

    def __get_actual_class(self):
        return self.source_data.uniq_values()[self.__class_select_button.active]

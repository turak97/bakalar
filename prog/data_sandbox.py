
from basic_sublayouts import SubLayout

from bokeh.models import PointDrawTool, Div, RadioButtonGroup, TextInput, \
    Button, Select, Slider
from bokeh import events
from bokeh.layouts import row, column

from random import randint

from in_n_out import save_source
import data_gen as dg

from constants import DENS_INPUT_DEF_VAL, CLUSTER_SIZE_DEF, CLUSTER_VOL_DEF, CLUSTERS_COUNT_DEF, MAX_CLUSTERS, \
    SAVED_DATASET_FILE_NAME, EMPTY_VALUE_COLOR, LASSO_SLIDER_END, LASSO_SLIDER_START, LASSO_SLIDER_STARTING_VAL, \
    LASSO_SLIDER_STEP

STANDARD_MODE = "Standard mode"
LASSO_APPEND = "Append with Lasso"

# classification data sandbox modes
GENERATE_NEW_CLUSTERS = "New clusters"


# TODO: at se to neodviji od uniqvalues
# TODO: APPROX

# TODO: bug: unexpected chovani pri odstraneni vsech bodu
# TODO: bug: points in dataset obcas zobrazuje o 1 mensi hodnotu, nez self.data.classification u BUGCHECKu
# TODO: bug: pocet trid nahore 3, pridavani clusteru s clusres count 6 spadne


class DataSandbox(SubLayout):
    def __init__(self, name, source_data):
        SubLayout.__init__(self, name, source_data)
        self.source_data.plot_source.on_change('data', self._plot_source_change)  # DataSandbox can update statistics

        self._fig.on_event(events.SelectionGeometry, self._lasso_add)

    """Methods for sublayout initialisation"""

    def _layout_init(self):
        fig_layout = self._init_figure()
        button_layout = self._init_button_layout()
        return row(fig_layout, button_layout)
    
    def _init_button_layout(self):
        basic_buttons = self._init_basic_buttons()

        self._generation_modes = {
            STANDARD_MODE: self._init_standard_option(),
            LASSO_APPEND: self._init_lasso_option()
        }
        self._add_special_modes()

        mode_button_labels = list(self._generation_modes.keys())
        mode_button_width = 120 * (len(mode_button_labels) + 1)
        self._points_generation_mode_button = RadioButtonGroup(
            labels=mode_button_labels, active=0, width=mode_button_width, width_policy="fixed")
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
        self._data_size_info = Div(text="Points in dataset: " + str(len(self.source_data.plot_source.data['x'])))

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
                                            step=LASSO_SLIDER_STEP, value=LASSO_SLIDER_STARTING_VAL)

        return column(lasso_general_info,
                      self._lasso_density_slider
                      )

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
        self._data_size_info.update(text="Points in dataset: " + str(len(new['x'])))

    def _lasso_add(self, event):
        if event.final and self._is_lasso_append_active():

            keys = event.geometry['x'].keys()  # 'x' and 'y' have the same keys which is the number of vertex
            vertices = []
            for key in keys:
                vertices.append((event.geometry['x'][key], event.geometry['y'][key]))

            self._append_to_source(vertices)

    def _save_dataset(self):
        self._info("Saving dataset...")
        abs_path = self.source_data.save_source(self._save_path.value)
        self._save_info.update(text="Saved in: " + abs_path)
        self._info("Saved in " + str(abs_path))
        self._info("Saving DONE")

    """Other methods"""

    def _append_to_source(self, vertices):
        pass

    def _get_lasso_density(self):
        return self._lasso_density_slider.value

    def _is_lasso_append_active(self):
        active_mode = self._points_generation_mode_button.labels[
            self._points_generation_mode_button.active]
        return active_mode == LASSO_APPEND


class RegressionDataSandbox(DataSandbox):
    def __init__(self, name, source_data):
        DataSandbox.__init__(self, name, source_data)

        move_circle = self._fig.circle('x', 'y', source=source_data.plot_source, size=7)
        point_draw_tool = PointDrawTool(renderers=[move_circle], empty_value=EMPTY_VALUE_COLOR, add=True)
        self._fig.add_tools(point_draw_tool)

    """Methods for sublayout initialisation"""

    def __aprox_poly_options(self):
        __aprox_general_info = Div(
            text="TODO")

        self.__aprox_density_slider = Slider(start=5, end=100, value=20, step=5)

        __aprox_options_info = Div(text="Aprox poly utilities: ", style={'font-size': '150%'})

        return column(__aprox_general_info,
                      __aprox_options_info,
                      self.__aprox_density_slider
                      )

    """Other methods"""

    def _append_to_source(self, vertices):
        density = self._get_lasso_density()
        self.source_data.append_polygon_data(vertices, density)


class ClassifierDataSandbox(DataSandbox):
    def __init__(self, name, source_data, class_select_button):
        DataSandbox.__init__(self, name, source_data)

        move_circle = self._fig.circle('x', 'y', color='color', source=source_data.plot_source, size=7)
        point_draw_tool = PointDrawTool(renderers=[move_circle], empty_value=EMPTY_VALUE_COLOR, add=True)
        self._fig.add_tools(point_draw_tool)

        self.__class_select_button = class_select_button

    def __del__(self):
        self.source_data.plot_source.remove_on_change('data', self._plot_source_change)  # removing trigger

    """Methods for sublayout initialisation"""

    def _add_special_modes(self):
        self._generation_modes[GENERATE_NEW_CLUSTERS] = self.__init_cluster_generating_options()

    def __init_lasso_options(self):
        __lasso_general_info = Div(
            text="Add a new cluster by selecting \"Lasso Select\" in the figure toolbar. "
                 "You can choose whether the class of cluster will be chosen by upwards selection (\"Seleted\") or "
                 "the classes of all points in the cluster will be chosen at random (\"All points at random\"). "
                 "Size of the generated cluster can be set either roughly or precisely. "
                 "If some points are not visible properly, click on \"Reset\" "
                 "in the figure toolbar for resetting the view.")

        self.__class_of_cluster_button = RadioButtonGroup(
            labels=["Selected", "All points at random"], active=0,
            width=220
        )
        class_of_cluster_text = Div(text="Class of new cluster: ")

        self.__lasso_point_density_button = RadioButtonGroup(
            labels=["auto", "±5", "±10", "±20", "±50", "±100"],
            active=0, width=300, width_policy="fixed"
        )
        self.__lasso_point_density_button.on_change('active', self.__lasso_density_button_trigger)

        self.__lasso_point_density_input = TextInput(value=DENS_INPUT_DEF_VAL, width=50)
        self.__lasso_point_density_input.on_change('value', self.__lasso_density_input_trigger)
        __lasso_options_info = Div(text="Lasso utilities: ", style={'font-size': '150%'})
        __lasso_density_info = Div(text="Density options:", style={'font-size': '120%'})
        __lasso_circa_info = Div(text="Circa: ")
        __lasso_exact_info = Div(text="or Precise: ")
        return column(__lasso_general_info,
                      __lasso_options_info,
                      row(class_of_cluster_text, self.__class_of_cluster_button),
                      __lasso_density_info,
                      column(row(__lasso_circa_info, self.__lasso_point_density_button),
                             row(__lasso_exact_info, self.__lasso_point_density_input))
                      )

    def __init_cluster_generating_options(self):
        __generate_general_info = Div(
            text="Automaticly generate a new clusters. "
                 "You can choose whether the new cluster should replace data or replace them. "
                 "\"Clusters count\" option determines the number of cluster (each has it onw class), "
                 "\"size\" sets the size of clusters and \"±\" sets the volatility of size.")

        __generate_info = Div(text="New clusters: ", style={'font-size': '150%'})

        self.__new_clusters_mode_button = RadioButtonGroup(
            labels=["Replace", "Append"], width=200, active=0, sizing_mode="fixed")
        __new_clusters_mode_text = Div(text="Clusters adding mode: ")

        self.__cluster_count_input = Select(title="Clusters count ", value=str(CLUSTERS_COUNT_DEF),
                                            options=[str(i) for i in range(1, MAX_CLUSTERS + 1)], width=80)
        self.__cluster_size_input = TextInput(title="size ", value=str(CLUSTER_SIZE_DEF), width=50)
        self.__cluster_plusminus_input = TextInput(title="±", value=str(CLUSTER_VOL_DEF), width=50)

        __generate_new_clusters_button = Button(label="Generate", button_type="primary")
        __generate_new_clusters_button.on_click(self.__generate_new_clusters)

        return column(__generate_general_info,
                      __generate_info,
                      row(__new_clusters_mode_text, self.__new_clusters_mode_button),
                      row(self.__cluster_count_input, self.__cluster_size_input, self.__cluster_plusminus_input),
                      __generate_new_clusters_button
                      )

    """Methods for interactive calling (called by on_change or on_click triggers)"""

    def __generate_new_clusters(self):
        self._info("Generating new dataset...")

        clusters_count, clusters_size, clusters_vol = self.__get_cluster_generating_params()

        x, y, classification = dg.cluster_data(x_interval=(0, 30), y_interval=(-10, 10),
                                               clusters=clusters_count,
                                               av_cluster_size=clusters_size,
                                               clust_size_vol=clusters_vol)

        if 0 == self.__new_clusters_mode_button.active:
            self.source_data.replace_data(x=x, y=y, classification=classification)
        else:
            self.source_data.append_data(x_new=x, y_new=y, classification_new=classification)

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
        clusters_count = int(self.__cluster_count_input.value)
        clusters_size = self.__get_int_set_error(
            text_input=self.__cluster_size_input, lowest_val=1
        )
        clusters_vol = self.__get_int_set_error(
            text_input=self.__cluster_plusminus_input, lowest_val=0
        )
        return clusters_count, clusters_size, clusters_vol

    def _append_to_source(self, vertices):
        density = self._get_lasso_density()
        cls = self.source_data.uniq_values()[self.__class_select_button.active]
        self.source_data.append_polygon_data(vertices, density, cls)

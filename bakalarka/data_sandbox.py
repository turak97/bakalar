
from basic_sublayouts import SubLayout

from bokeh.models import PointDrawTool, CheckboxButtonGroup, \
    Div, RadioButtonGroup, TextInput, Button, Select
from bokeh import events
from bokeh.layouts import row, column

from random import randint

from in_n_out import save_source
import data_gen as dg
from constants import DENS_INPUT_DEF_VAL, CLUSTER_SIZE_DEF, CLUSTER_VOL_DEF, CLUSTERS_COUNT_DEF, MAX_CLUSTERS, \
    SAVED_DATASET_FILE_NAME


# TODO: at se to neodviji od uniqvalues

# TODO: remove points s lasem?

# TODO: bug: unexpected chovani pri odstraneni vsech bodu
# TODO: bug: points in dataset obcas zobrazuje o 1 mensi hodnotu, nez self.data.classification u BUGCHECKu


class DataSandbox(SubLayout):
    def __init__(self, name, source_data, class_select_button):
        SubLayout.__init__(self, name, source_data)

        self.__class_select_button = class_select_button

        self.source_data.plot_source.on_change('data', self.__plot_source_change)  # DataSandbox can update statistics

        self._fig.on_event(events.SelectionGeometry, self.__lasso_update)

    def __del__(self):
        self.source_data.plot_source.remove_on_change('data', self.__plot_source_change)  # removing trigger

    def __lasso_update(self, event):
        if event.final and 0 == self.__points_generation_mode.active:

            keys = event.geometry['x'].keys()  # 'x' and 'y' have the same keys which is the number of vertex
            vertices = []
            for key in keys:
                vertices.append((event.geometry['x'][key], event.geometry['y'][key]))

            cluster_size = self.__get_cluster_size()
            x_new, y_new = dg.polygon_data(vertices, cluster_size)
            classification_new = self.__generate_classes(len(x_new))
            self.source_data.append_data(x_new, y_new, classification_new)

    def __plot_source_change(self, attr, old, new):
        self.__data_size_info.update(text="Points in dataset: " + str(len(new['x'])))

    def _layout_init(self):
        fig_layout = self._init_figure()
        button_layout = self._init_button_layout()
        return row(fig_layout, button_layout)

    def _init_button_layout(self):
        self.__data_size_info = Div(text="Points in dataset: " + str(len(self.source_data.plot_source.data['x'])))

        self.__save_button = Button(label="Save dataset", button_type="primary")
        self.__save_button.on_click(self.__save_dataset)

        self.__save_path = TextInput(value=SAVED_DATASET_FILE_NAME, title="Name of the file:")
        self.__save_info = Div(text="", style={'font-size': '75%'}, width_policy="fixed")
        save_group = column(self.__save_path, self.__save_button, self.__save_info)

        mode_button_labels = ["Lasso", "Automatic generation"]
        mode_button_width = 120 * (len(mode_button_labels) + 1)
        self.__points_generation_mode = RadioButtonGroup(
            labels=mode_button_labels, active=0, width=mode_button_width, width_policy="fixed")
        self.__points_generation_mode.on_change('active', self.__points_generation_mode_trigger)
        mode_text = Div(text="Data sandbox mode: ", style={'font-size': '150%'})

        self.__generation_modes = {
            0: self.__init_lasso_options(),
            1: self.__init_cluster_generating_options()
        }

        active_mode = self.__generation_modes[self.__points_generation_mode.active]
        return column(self.__data_size_info,
                      save_group,
                      mode_text,
                      self.__points_generation_mode,
                      active_mode,
                      )

    def __save_dataset(self):
        self._info("Saving dataset...")
        abs_path = self.source_data.save_source(self.__save_path.value)
        self.__save_info.update(text="Saved in: " + abs_path)
        self._info("Saved in " + str(abs_path))
        self._info("Saving DONE")

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

    def __points_generation_mode_trigger(self, attr, old, new):
        new_mode = self.__generation_modes[new]
        self.layout.children[1].children[4] = new_mode

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

    def __generate_new_clusters(self):
        self._info("Generating new dataset...")

        clusters_count, clusters_size, clusters_vol = self.__get_cluster_generating_params()
        print(self.__get_cluster_generating_params())

        x, y, classification = dg.cluster_data(x_interval=(0, 30), y_interval=(-10, 10),
                                               clusters=clusters_count,
                                               av_cluster_size=clusters_size,
                                               clust_size_vol=clusters_vol)

        if 0 == self.__new_clusters_mode_button.active:
            self.source_data.replace_data(x=x, y=y, classification=classification)
        else:
            self.source_data.append_data(x_new=x, y_new=y, classification_new=classification)

        self._info("Generating new dataset DONE")

    def __get_cluster_size(self):
        if self.__lasso_point_density_input.value != DENS_INPUT_DEF_VAL:
            # density input provides positive integer
            return int(self.__lasso_point_density_input.value)

        # TODO: exception self.__lasso_point_density_button.active is None
        raw_cluster_size = self.__lasso_point_density_button.labels[self.__lasso_point_density_button.active][1:]
        if raw_cluster_size == "uto":
            return -1
        precise_cluster_size = int(raw_cluster_size)
        volatility = int(precise_cluster_size/2)
        return precise_cluster_size + randint(-volatility, volatility)

    def __lasso_density_button_trigger(self, attr, old, new):
        if self.__lasso_point_density_input.value != DENS_INPUT_DEF_VAL:
            # make sure density input is not providing any value
            self.__lasso_point_density_input.remove_on_change('value', self.__lasso_density_input_trigger)
            self.__lasso_point_density_input.update(value=DENS_INPUT_DEF_VAL)
            self.__lasso_point_density_input.on_change('value', self.__lasso_density_input_trigger)

    def __lasso_density_input_trigger(self, attr, old, new):
        """
        this method provides that density_input widget value is ALWAYS either set to default value
        or provides positive integer
        """
        if self.__lasso_point_density_button.active is not None:  # make sure density button is not providing any value
            self.__lasso_point_density_button.remove_on_change('active', self.__lasso_density_button_trigger)
            self.__lasso_point_density_button.update(active=None)
            self.__lasso_point_density_button.on_change('active', self.__lasso_density_button_trigger)

        self.__lasso_point_density_input.remove_on_change('value', self.__lasso_density_input_trigger)
        potential_cluster_size = 0
        try:
            potential_cluster_size = int(self.__lasso_point_density_input.value)
        except ValueError:
            self.__lasso_point_density_input.update(value=DENS_INPUT_DEF_VAL)
        if potential_cluster_size < 1:
            self.__lasso_point_density_input.update(value=DENS_INPUT_DEF_VAL)
        self.__lasso_point_density_input.on_change('value', self.__lasso_density_input_trigger)

    def __generate_classes(self, length):
        if self.__class_of_cluster_button.active == 1:  # if Random classes active
            return dg.classify(length, self.source_data.uniq_values())
        return [self.source_data.uniq_values()[self.__class_select_button.active] for _ in range(length)]

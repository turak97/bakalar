
from Layout import SubLayout
from bokeh.models import PointDrawTool, CheckboxButtonGroup, \
    Div, RadioButtonGroup, TextInput, Button, Select
from bokeh import events
from bokeh.layouts import row, column

from random import randint

import data_gen as dg
from constants import DENS_INPUT_DEF_VAL, CLUSTER_SIZE_DEF, CLUSTER_VOL_DEF, CLUSTERS_COUNT_DEF, MAX_CLUSTERS


# TODO: at se to neodviji od uniqvalues

# TODO: random moznost u vyberu trid
# TODO: remove points s lasem?

# TODO: bug: unexpected chovani pri odstraneni vsech bodu
# TODO: bug: points in dataset obcas zobrazuje o 1 mensi hodnotu, nez self.data.classification u BUGCHECKu

# TODO: vyclenit to z nabidky samostatny toggle button

# TODO: automaticky a manualni rezim pridavani/ preklikavat/ podle toho se zorazi dole tlacitka
# TODO: u manualniho rezimu popridavat vysvetlivky

# TODO: save as csv v DataSandobxu, pres Div dat vedet, co se deje, v text. poli moznost zvolit nazev

class DataSandbox(SubLayout):
    def __init__(self, name, plot_info, class_select_button):
        SubLayout.__init__(self, name, plot_info)

        self.__class_select_button = class_select_button

        move_circle = self._fig.circle('x', 'y', color='color', source=plot_info.plot_source, size=7)
        point_draw_tool = PointDrawTool(renderers=[move_circle], empty_value='black', add=True)
        self._fig.add_tools(point_draw_tool)

        self.plot_info.plot_source.on_change('data', self.__plot_source_change)  # DataSandbox can update statistics

        self._fig.on_event(events.SelectionGeometry, self.__lasso_update)

    def __del__(self):
        self.plot_info.plot_source.remove_on_change('data', self.__plot_source_change)  # removing trigger

    def _layout_init(self):
        return row(self._fig, row())

    def __lasso_update(self, event):
        if event.final and 0 in self.__lasso_button.active:

            keys = event.geometry['x'].keys()  # 'x' and 'y' have the same keys which is the number of vertex
            vertices = []
            for key in keys:
                vertices.append((event.geometry['x'][key], event.geometry['y'][key]))

            cluster_size = self.__get_cluster_size()
            x_new, y_new = dg.polygon_data(vertices, cluster_size)
            classification_new = self.__generate_classes(len(x_new))
            self.plot_info.append_data(x_new, y_new, classification_new)

    def __plot_source_change(self, attr, old, new):
        self.__data_size_info.update(text="Points in dataset: " + str(len(new['x'])))

    def _init_button_layout(self):
        self.__data_size_info = Div(text="Points in dataset: " + str(len(self.plot_info.plot_source.data['x'])))

        cluster_generating_options = self.__init_cluster_generating_options()
        lasso_options = self.__init_lasso_options()
        self.layout.children[1] = column(self.__data_size_info,
                                         cluster_generating_options,
                                         lasso_options
                                         )

    def __init_lasso_options(self):
        self.__lasso_button = CheckboxButtonGroup(
            labels=["Generate points with lasso", "Random classes"],
            active=[]
        )
        self.__lasso_point_density_button = RadioButtonGroup(
            labels=["auto", "±5", "±10", "±20", "±50", "±100"],
            active=0, width=300, width_policy="fixed"
        )
        self.__lasso_point_density_button.on_change('active', self.__lasso_density_button_trigger)

        self.__lasso_point_density_input = TextInput(value=DENS_INPUT_DEF_VAL, width=50)
        self.__lasso_point_density_input.on_change('value', self.__lasso_density_input_trigger)
        __lasso_options_info = Div(text="Lasso utilities: ", style={'font-size': '150%'})
        __lasso_density_info = Div(text="Density:", style={'font-size': '120%'})
        __lasso_circa_info = Div(text="Circa: ")
        __lasso_exact_info = Div(text="or Precise: ")
        return column(__lasso_options_info,
                      row(self.__lasso_button),
                      __lasso_density_info,
                      column(row(__lasso_circa_info, self.__lasso_point_density_button),
                             row(__lasso_exact_info, self.__lasso_point_density_input))
                      )

    def __init_cluster_generating_options(self):
        __generate_info = Div(text="Generate new dataset: ", style={'font-size': '150%'})

        self.__cluster_count_input = Select(title="Clusters count ", value=str(CLUSTERS_COUNT_DEF),
                                            options=[str(i) for i in range(1, MAX_CLUSTERS + 1)], width=80)
        self.__cluster_size_input = TextInput(title="size ", value=str(CLUSTER_SIZE_DEF), width=50)
        self.__cluster_plusminus_input = TextInput(title="±", value=str(CLUSTER_VOL_DEF), width=50)

        __generate_new_dataset_button = Button(label="Generate", button_type="primary")
        __generate_new_dataset_button.on_click(self.__generate_new_dataset)

        return column(__generate_info,
                      row(self.__cluster_count_input, self.__cluster_size_input, self.__cluster_plusminus_input),
                      __generate_new_dataset_button
                      )

    @staticmethod
    def __get_int_set_error(text_input, lowest_val):
        """
        returns POSITIVE INTEGER from text_input, if the value is different, returns "default" instead
        """
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

    def __generate_new_dataset(self):
        self._info("Generating new dataset...")

        clusters_count, clusters_size, clusters_vol = self.__get_cluster_generating_params()
        print(self.__get_cluster_generating_params())

        x, y, classification = dg.cluster_data(x_interval=(0, 30), y_interval=(-10, 10),
                                               clusters=clusters_count,
                                               av_cluster_size=clusters_size,
                                               clust_size_vol=clusters_vol)
        self.plot_info.replace_data(x=x, y=y, classification=classification)

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
        if 1 in self.__lasso_button.active:  # if Random classes active
            return dg.classify(length, self.plot_info.uniq_values())
        return [self.plot_info.uniq_values()[self.__class_select_button.active] for _ in range(length)]

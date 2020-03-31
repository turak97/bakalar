
import numpy as np
from bokeh.models import RadioGroup, CheckboxButtonGroup, ColorBar, RadioButtonGroup, ColorPicker
from bokeh.models.widgets import Dropdown
from bokeh.layouts import row, column
import plotting_utilities as pu
from bokeh.models import LassoSelectTool, Div

from bokeh.plotting import figure

# TODO: promennou na umisteni RadioGroupy a SubLayoutu v Layoutu
# TODO: pouze jeden logovaci vypis
# TODO: nabidka palet
# TODO: Layout bude tvorit PlotInfo v konstruktoru?


def concat(x, y):
    return np.array([[x[i], y[i]] for i in range(len(x))])


class Data:
    def __init__(self, x_data, y_data, classification):
        self.x_data = x_data
        self.y_data = y_data
        self.cls_X = concat(x_data, y_data)
        self.classification = classification

    def push_new_points(self, source, c):
        new_from_i = len(source.data['x']) - 1
        source_len = len(source.data['x'])

        to_append_x = np.array(source.data['x'][new_from_i:])
        to_append_y = np.array(source.data['y'][new_from_i:])
        to_append_class = [np.int(c)] * (source_len - new_from_i)

        self.x_data = np.append(self.x_data, to_append_x)
        self.y_data = np.append(self.y_data, to_append_y)
        self.cls_X = concat(self.x_data, self.y_data)
        self.classification = np.append(self.classification, to_append_class)

    def replace_data(self, source):
        self.x_data = np.array(source.data['x'])
        self.y_data = np.array(source.data['y'])
        self.cls_X = concat(self.x_data, self.y_data)
        self.classification = np.array(source.data['classification'])


class Layout:
    def __init__(self, data, plot_source, plot_info):
        self.plot_info = plot_info
        self.data = data
        self.plot_source = plot_source
        self.plot_source.on_change('data', self.__data_change)

        self.__model_selection_init()
        self.__info = Div(text="Hello world!")
        self.__class_selection_init()

        self.__sub_layouts = []

        self.layout = column(column(self.__dropdown,
                                    self.__options,
                                    self.__info),
                             row(self.__radio_group,
                                 row()))  # row() is a place for added SubLayouts
        self.__sl0, self.__sl1 = 1, 1  # SubLayouts position

    def __model_selection_init(self):
        """Creates __dropdown and __options attribute, updates immediate_update in plot_info"""
        menu = [("Polynomial regression", "reg.polynomial"), None, ("SVC linear", "clas.svc_linear"),
                ("SVC Polynomial", "clas.svc_poly"), ("SVC Sigmoid", "clas.svc_sigmoid"),
                ("K nearest neighbours", "clas.knn"), ("Naive Bayes", "clas.bayes"),
                ("Neural classification", "clas.neural")]

        self.__dropdown = Dropdown(label="+ add model", button_type="primary", menu=menu)
        self.__dropdown.on_click(self.__new_sub_layout)

        self.__options = CheckboxButtonGroup(labels=["Immediate update"], width=150,
                                           active=[0])
        self.__options.on_change('active', self.__options_change)
        self.plot_info.immediate_update = 0 in self.__options.active  # is Immediate update active?

    def __class_selection_init(self):
        self.__radio_group = RadioGroup(
            labels=["class 0", "class 1", "class 2"], active=0,
            width=100
        )

    def __options_change(self, attr, old, new):
        if 0 in new:  # immediate update active
            self.plot_info.immediate_update = True
        else:
            self.plot_info.immediate_update = False

    def __new_sub_layout(self, value):
        # model_res_str is for resolution to sci kit model
        # model_name is a fancy name for CheckBoxButtonGroup
        model_res, model_name = value.item, pu.find_first(self.__dropdown.menu, value.item)
        self._info("Adding a new " + model_name + " plot...")

        sub_layout = pu.resolution(model=model_res, name=model_name,
                                   data=self.data, plot_info=self.plot_info)
        self.__sub_layouts.append(sub_layout)
        self.layout.children[self.__sl0].children[self.__sl1] = pu.list_to_row(self.__sub_layouts)
        self.__update_checkbox_column()

        self._info("Done")

    def __del_sub_layout(self, attr, old, new):
        self._info("Deleting layout...")
        removed_i = new[0]  # list(set(old) - set(new))[0]
        del self.__sub_layouts[removed_i]
        self.layout.children[self.__sl0].children[self.__sl1] = pu.list_to_row(self.__sub_layouts)
        self.__update_checkbox_column()
        self._info("Done")

    def __update_checkbox_column(self):
        labels = [sub_lay.name for sub_lay in self.__sub_layouts]
        checkbox_button = CheckboxButtonGroup(labels=labels, default_size=200*len(self.__sub_layouts),
                                              active=[])
        checkbox_button.on_change('active', self.__del_sub_layout)
        self.layout.children[0].children[0] = row(checkbox_button, self.__dropdown)

    def __data_change(self, attr, old, new):
        self._info("Updating data...")
        if len(old['x']) < len(new['x']):  # new point/s added
            self.plot_info.update_color(new_class=self.__radio_group.active,
                                        f=self.__data_change, new_i=len(old['x']))
            self.data.push_new_points(self.plot_source, self.__radio_group.active)
        else:
            self.data.replace_data(self.plot_source)

        if self.plot_info.immediate_update:
            for lay in self.__sub_layouts:
                self._info("Updating figure " + lay.name + "...")
                lay.figure_update()
        self._info("Done")

    def _info(self, message):
        self.__info.text = str(message)


class SubLayout:
    def __init__(self, name, data, plot_info):
        self.name = name
        self.data = data
        self.plot_info = plot_info
        self.fig = figure(tools="pan,wheel_zoom,save,reset,box_zoom")
        self.fig.add_tools(LassoSelectTool())
        self.info = Div(text="Hello world!")
        # last one row() is for children needs changed in _init_button_layout
        self.layout = column(self.info, self.fig, row())

    def figure_update(self):
        pass

    def _init_button_layout(self):
        pass

    def _info(self, message):
        self.info.text = str(message)

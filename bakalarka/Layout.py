
import numpy as np
from bokeh.models import CheckboxButtonGroup, RadioButtonGroup, ColorPicker, Button
from bokeh.models.widgets import Dropdown
from bokeh.layouts import row, column
import plotting_utilities as pu
from bokeh.models import LassoSelectTool, Div

from bokeh.plotting import figure

# TODO: pekneji osetrit picker (aby nenastal pripad dvou stejnych barev)
# TODO: pouze jeden logovaci vypis
# TODO: Layout bude tvorit PlotInfo v konstruktoru?


CLASS_BUTTON_WIDTH = 100
MAX_CLASS_NAME_LENGTH = 8


def concat(x, y):
    return np.array([[x[i], y[i]] for i in range(len(x))])


class Data:
    def __init__(self, x_data, y_data, classification, classes_count):
        self.x_data = x_data
        self.y_data = y_data
        self.cls_X = concat(x_data, y_data)
        self.classification = classification
        self.classes_count = classes_count

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
    def __init__(self, data, plot_info):
        self.plot_info = plot_info
        self.data = data
        self.plot_info.plot_source.on_change('data', self.__data_change)

        self.__sub_layouts = []

        options = self.__model_selection_init()
        self.__info = Div(text="Hello world!")
        class_selection = self.__class_selection_init()

        self.layout = column(column(self.__info,
                                    self.__dropdown,
                                    row(options, class_selection)
                                    ),
                             row(
                                 row()))  # row() is a place for added SubLayouts
        self.__sl0, self.__sl1 = 1, 0  # SubLayouts position
        self.__dp0, self.__dp1 = 0, 1  # DropDown position
        self.__cs0, self.__cs1, self.__cs2 = 0, 2, 1  # __class_selection position

    def __model_selection_init(self):
        """Creates __dropdown, __options and __fit_all attribute, updates immediate_update in plot_info
        returns column of options
        """
        menu = [("Polynomial regression", "reg.polynomial"), None,
                ("SVM classification", "cls.svm"),
                ("K nearest neighbours", "cls.knn"), ("Naive Bayes (Gaussian)", "cls.bayes"),
                ("Neural classification", "cls.neural")]

        self.__dropdown = Dropdown(label="+ add model", button_type="primary", menu=menu,
                                   width_policy="fixed", width=170
                                   )
        self.__dropdown.on_click(self.__new_sub_layout)

        self.__fit_all = Button(label="Fit all", button_type="success", width=150)
        self.__fit_all.on_click(self.__update_sublayouts)

        self.__options = CheckboxButtonGroup(labels=["Immediate update"], width=150,
                                             active=[]  # affects immediate_update in plot_info
                                             )
        self.__options.on_change('active', self.__options_change)
        self.plot_info.immediate_update = False

        return column(self.__options, self.__fit_all)

    def __class_selection_init(self):
        classes_count = len(self.plot_info.uniq_values)
        try:
            old_active = self.__class_select_button.active
        except AttributeError:
            old_active = 0

        button_width = CLASS_BUTTON_WIDTH * self.data.classes_count
        picker_width = CLASS_BUTTON_WIDTH - self.__normalise_picker_width(classes_count)
        self.__color_pickers = [ColorPicker(
            title="",
            width=picker_width, width_policy="fixed",
            color=self.plot_info.palette[i]
        ) for i in range(self.data.classes_count)]
        for picker in self.__color_pickers:
            picker.on_change('color', self.__change_color)
        self.__class_select_button = RadioButtonGroup(
            labels=self.__normalise_uniq_values(self.plot_info.uniq_values),
            active=old_active,
            width=button_width, width_policy="fixed"
        )

        if classes_count > 9:
            return column(row(self.__color_picker),
                          row(self.__class_select_button))

        self.__new_class_button = Button(label="+ new class", button_type="primary",
                                         width=CLASS_BUTTON_WIDTH, width_policy="fixed"
                                         )
        self.__new_class_button.on_click(self.__new_class)
        return column(row(self.__color_pickers),
                      row(self.__class_select_button, self.__new_class_button))

    @staticmethod
    def __normalise_picker_width(classes_count):
        if classes_count <= 2:
            return 5
        if classes_count <= 4:
            return 7
        if classes_count <= 6:
            return 8
        return 9

    @staticmethod
    def __normalise_uniq_values(uniq_values):
        def normalise_val(val):
            if len(val) == MAX_CLASS_NAME_LENGTH:
                return val
            if len(val) < MAX_CLASS_NAME_LENGTH:
                while len(val) < MAX_CLASS_NAME_LENGTH:
                    if len(val) % 2 == 0:
                        val = val + u'\u00A0'
                    else:
                        val = u'\u00A0' + val
                return val
            # now len(val) > MAX_CLASS_NAME_LENGTH
            return val[:3] + ".." + val[(len(val) - 3):]

        normalised = []
        for uq in uniq_values:
            normalised.append(normalise_val(uq))
        return normalised

    def __change_color(self, attr, old, new):
        # two colors check
        if new == "#000000":
            for picker in self.__color_pickers:
                if picker.color == "#000000":
                    picker.color = old
                    return

        self.plot_info.replace_color(
            old_color=old, new_color=new, f=self.__data_change
        )

        for layout in self.__sub_layouts:
            layout.update_renderer_colors()

    def __new_class(self):
        self.data.classes_count += 1
        self.plot_info.add_new_color(
            class_name=self.__new_name(prev=self.plot_info.uniq_values[-1]))

        __class_selection = self.__class_selection_init()
        self.layout.children[self.__cs0].children[self.__cs1].children[self.__cs2] = __class_selection

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
        self.layout.children[self.__dp0].children[self.__dp1] = row(checkbox_button, self.__dropdown)

    def __data_change(self, attr, old, new):
        self._info("Updating data...")
        if len(old['x']) < len(new['x']):  # new point/s added
            new_class = self.plot_info.uniq_values[self.__class_select_button.active]
            self.plot_info.update_color_newly_added(new_class,
                                                    new_i=len(old['x']), f=self.__data_change)
            self.data.push_new_points(self.plot_info.plot_source, self.__class_select_button.active)
        else:
            self.data.replace_data(self.plot_info.plot_source)

        if self.plot_info.immediate_update:
            self.__update_sublayouts()
        self._info("Done")

    def __update_sublayouts(self):
        self.__fit_all.update(disabled=True)
        for lay in self.__sub_layouts:
            self._info("Updating figure " + lay.name + "...")
            lay.refit()
        self.__fit_all.update(disabled=False)

    def _info(self, message):
        self.__info.text = str(message)

    def __new_name(self, prev):
        if prev.isdigit():
            return str(int(prev) + 1)
        return "extra " + str(len(self.plot_info.uniq_values))


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

    def refit(self):
        pass

    def _init_button_layout(self):
        pass

    def _info(self, message):
        self.info.text = str(message)


from bokeh.models import CheckboxButtonGroup, RadioButtonGroup, ColorPicker, Button, Div
from bokeh.models.widgets import Dropdown
from bokeh.layouts import row, column

import sublayout_resolution as sr
from constants import CLASS_SELECT_BUTTON_WIDTH, MAX_CLASS_NAME_LENGTH, EMPTY_VALUE_COLOR


# TODO: moznost prepnout mezi klasifikacni, regresni nebo obema verzema appky

# TODO: zkusit lip vyresit memeni children v layoutech (row, column)

# TODO: pekneji osetrit picker (aby nenastal pripad dvou stejnych barev), reseni: zjistit, ktery picker triggnul funkci?
# TODO: uniq values dat dokupy

# TODO: Layout bude tvorit PlotInfo v konstruktoru?


class GeneralLayout:
    def __init__(self, plot_info, log=True):
        self.__log = log  # TODO: dodelat

        self.plot_info = plot_info
        self.plot_info.set_plot_source_trigger(self.__data_change)

        self.__sub_layouts = []

        model_selection = self.__model_selection_init()
        fit_all = self.__fit_all_init()
        class_selection = self.__class_selection_init()
        data_sandbox_button = self.__sand_box_button_init()

        general_info = Div(
            text="Add a new figure by selecting one through clicking on \"+ add model\". "
                 "Figure can be deleted by clicking on it in the button group. "
                 "You can add a new/move/delete points in the dataset by selecting \"Point Draw Tool\" "
                 "in the figure toolbar. Then choose a class and click anywhere into the figure. You can select "
                 "more points by holding \"shift\" while selecting and delete them by with \"backspace\". "
                 "For more manipulating options with dataset click on \"Data Sandbox\". You can deactivate it by "
                 "clicking on the option again. Every figure can be updated immediately after adding/moving/deleting"
                 " points by selecting \"Immediate update\" bellow the figure (this comes handy with fast algorithms "
                 "as SVM or Bayes).",
            width=1000
        )

        self.layout = column(column(general_info,
                                    row(class_selection),
                                    row(data_sandbox_button, fit_all, model_selection)
                                    ),
                             row(row(),  # this is a place for data sandbox
                                 row()))  # this is a place for added SubLayouts
        self.__sb1, self.__sb2 = 1, 0  # data sandbox position
        self.__sl0, self.__sl1 = 1, 1  # SubLayouts position
        self.__dp0, self.__dp1, self.__dp2 = 0, 2, 2  # DropDown (model_selection) position
        self.__cs0, self.__cs1, self.__cs2 = 0, 1, 0  # __class_selection position

    def __sand_box_button_init(self):
        """Initialise button for data sandbox activation"""
        self.__sandbox_button = CheckboxButtonGroup(labels=["Data Sandbox"], active=None,
                                                    width=150, width_policy="fixed")
        self.__sandbox_button.on_change('active', self.__data_sandbox_trigger)

        return self.__sandbox_button

    def __fit_all_init(self):
        """Initialise menu buttons and sets triggers on them
        return column of buttons"""

        self.__fit_all = Button(label="Fit all", button_type="success", width=150)
        self.__fit_all.on_click(self.__update_all_sublayouts)

        return self.__fit_all

    def __model_selection_init(self):
        """Initialise dropdown button for choosing model
        return __dropdown
        """
        menu = [("Polynomial regression", "reg.polynomial"), None,
                ("SVM classification", "cls.svm"),
                ("K nearest neighbours", "cls.knn"), ("Naive Bayes (Gaussian)", "cls.bayes"),
                ("Neural classification", "cls.neural")]

        self.__dropdown = Dropdown(label="+ add model", button_type="primary", menu=menu,
                                   width_policy="fixed", width=170
                                   )
        self.__dropdown.on_click(self.__new_sub_layout)

        return self.__dropdown

    def __class_selection_init(self):
        classes_count = len(self.plot_info.uniq_values())
        try:
            old_active = self.__class_select_button.active
        except AttributeError:
            old_active = 0

        button_width = CLASS_SELECT_BUTTON_WIDTH * classes_count
        picker_width = CLASS_SELECT_BUTTON_WIDTH - self.__normalise_picker_width(classes_count)
        self.__color_pickers = [ColorPicker(
            title="",
            width=picker_width, width_policy="fixed",
            color=self.plot_info.palette[i]
        ) for i in range(classes_count)]
        for picker in self.__color_pickers:
            picker.on_change('color', self.__change_color)
        self.__class_select_button = RadioButtonGroup(
            labels=self.__normalise_uniq_values(self.plot_info.uniq_values()),
            active=old_active,
            width=button_width, width_policy="fixed"
        )

        if classes_count > 9:
            return column(row(self.__color_pickers),
                          row(self.__class_select_button))

        self.__new_class_button = Button(label="+ new class", button_type="primary",
                                         width=CLASS_SELECT_BUTTON_WIDTH, width_policy="fixed"
                                         )
        self.__new_class_button.on_click(self.__new_class)
        return column(row(self.__color_pickers),
                      row(self.__class_select_button, self.__new_class_button))

    def __class_selection_update(self):
        classes_count = len(self.plot_info.uniq_values())
        picker_width = CLASS_SELECT_BUTTON_WIDTH - self.__normalise_picker_width(classes_count)
        button_width = CLASS_SELECT_BUTTON_WIDTH * classes_count

        self.__color_pickers.append(
            ColorPicker(
                title="",
                width=picker_width, width_policy="fixed",
                color=self.plot_info.palette[classes_count - 1]
            )
        )
        self.__color_pickers[-1].on_change('color', self.__change_color)
        self.layout.children[self.__cs0].children[self.__cs1].children[self.__cs2].children[0] \
            = row(self.__color_pickers)

        new_labels = self.__normalise_uniq_values(self.plot_info.uniq_values())
        self.__class_select_button.update(labels=new_labels,
                                          width=button_width)

    def __data_sandbox_trigger(self, attr, old, new):
        if 0 in new:  # sandbox button was activated
            sandbox = sr.data_sandbox(name="Data Sandbox", plot_info=self.plot_info,
                                      class_select_button=self.__class_select_button)
            self.layout.children[self.__sb1].children[self.__sb2] = sandbox.layout
        else:
            self.layout.children[self.__sb1].children[self.__sb2] = row()

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

        if new == "#000000":  # two colors check
            for picker in self.__color_pickers:
                if picker.color == "#000000":
                    picker.color = old
                    return

        self.plot_info.replace_color(
            old_color=old, new_color=new
        )

        for sub_lay in self.__sub_layouts:
            sub_lay.update_renderer_colors()

    def __new_class(self):
        self.plot_info.add_new_color(
            class_name=self.__new_name(prev=self.plot_info.uniq_values()[-1])
        )

        self.__class_selection_update()
        # __class_selection = self.__class_selection_init()
        # self.layout.children[self.__cs0].children[self.__cs1].children[self.__cs2] = __class_selection

    def __new_sub_layout(self, value):
        # model_res_str is for resolution to sci kit model
        # model_name is a fancy name for CheckBoxButtonGroup
        model_res, model_name = value.item, GeneralLayout.__find_first(self.__dropdown.menu, value.item)
        self._info("Adding a new " + model_name + " plot...")

        sub_layout = sr.resolution(model=model_res, name=model_name,
                                   plot_info=self.plot_info)

        self.__sub_layouts.append(sub_layout)

        new_children = row([lay.layout for lay in self.__sub_layouts])
        self.layout.children[self.__sl0].children[self.__sl1] = new_children
        self.__update_checkbox_column()

        self._info("New plot DONE")

    @staticmethod
    def __find_first(tuple_list, second):
        """find first occurrence of second in tuple_list and returns first element"""
        for t1, t2 in filter(lambda x: x is not None, tuple_list):
            if t2 == second:
                return t1
        return None

    def __del_sub_layout(self, attr, old, new):
        self._info("Deleting layout...")
        removed_i = new[0]
        del self.__sub_layouts[removed_i]

        new_children = row([lay.layout for lay in self.__sub_layouts])
        self.layout.children[self.__sl0].children[self.__sl1] = new_children
        self.__update_checkbox_column()
        self._info("Deleting DONE")

    def __update_checkbox_column(self):
        labels = [sub_lay.name for sub_lay in self.__sub_layouts]
        checkbox_button = CheckboxButtonGroup(labels=labels, default_size=200*len(self.__sub_layouts),
                                              active=[])
        checkbox_button.on_change('active', self.__del_sub_layout)
        self.layout.children[self.__dp0].children[self.__dp1].children[self.__dp2] = \
            row(checkbox_button, self.__dropdown)

    def __data_change(self, attr, old, new):
        self._info("Updating data...")

        if len(old['x']) < len(new['x']):

            if new['color'][-1] == EMPTY_VALUE_COLOR:
                new_class = self.plot_info.uniq_values()[self.__class_select_button.active]
                self.plot_info.update_color_newly_added(new_class,
                                                        new_i=len(old['x']))

        self._info("Updating data DONE")

        self.__update_immediate_sublayouts()

    def __update_immediate_sublayouts(self):
        sub_layouts_to_update = [sub_lay for sub_lay in self.__sub_layouts if sub_lay.immediate_update()]
        if len(sub_layouts_to_update) == 0:
            return

        self._info("Immediate update of sublayouts...")

        for sub_lay in sub_layouts_to_update:
            self._info("Updating figure " + sub_lay.name + "...")
            sub_lay.refit()

        self._info("Updating sublayouts DONE")

    def __update_all_sublayouts(self):
        self._info("Updating all sublayouts")

        for sub_lay in self.__sub_layouts:
            self._info("Updating figure " + sub_lay.name + "...")
            sub_lay.refit()

        self._info("Updating DONE")

    @staticmethod
    def _info(message):
        print("Top layout: " + message)

    def __new_name(self, prev):
        if prev.isdigit():
            return str(int(prev) + 1)
        return "extra " + str(len(self.plot_info.uniq_values()))


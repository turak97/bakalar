
from bokeh.models import CheckboxButtonGroup, RadioButtonGroup, ColorPicker, Button, Div
from bokeh.models.widgets import Dropdown
from bokeh.layouts import row, column

import sublayout_resolution as sr
from constants import EMPTY_VALUE_COLOR
from models import REG_MODELS, CLS_MODELS

# used in general_layout.py
CLASS_SELECT_BUTTON_WIDTH = 100  # in pixels, default: 100
MAX_CLASS_NAME_LENGTH = 8  # default: 8


class GeneralLayout:
    def __init__(self, source_data, log=True):
        self._log = log  # TODO: dodelat

        self.source_data = source_data
        self.source_data.set_plot_source_trigger(self._data_change)

        self._sub_layouts = []

        self._layout_init()

        self._data_sandbox = self._create_data_sandbox()

    """Methods for layout initialisation"""

    def _layout_init(self):
        self.layout = row()

    def _sand_box_button_init(self):
        """Initialise button for data sandbox activation"""
        self._sandbox_button = CheckboxButtonGroup(labels=["Data Sandbox"], active=None,
                                                   width=150, width_policy="fixed")
        self._sandbox_button.on_change('active', self._data_sandbox_trigger)

        return self._sandbox_button

    def _fit_all_init(self):
        """Initialise menu buttons and sets triggers on them
        return column of buttons"""

        self._fit_all = Button(label="Fit all", button_type="success", width=150)
        self._fit_all.on_click(self._update_all_sublayouts)

        return self._fit_all

    @staticmethod
    def _menu_init():
        pass

    def _model_selection_init(self):
        """Initialise dropdown button for choosing model
        return _dropdown
        """
        menu = self._menu_init()

        self._dropdown = Dropdown(label="+ add model", button_type="primary", menu=menu,
                                  width_policy="fixed", width=170
                                  )
        self._dropdown.on_click(self._new_sub_layout)

        return self._dropdown

    """Methods for interactive calling (called by on_change or on_click triggers)"""

    def _data_sandbox_trigger(self, attr, old, new):
        if 0 in new:  # sandbox button was activated
            sandbox = self._data_sandbox
            self.layout.children[self._sb1].children[self._sb2] = sandbox.layout
        else:
            self.layout.children[self._sb1].children[self._sb2] = row()

    def _new_sub_layout(self, value):
        model_name = value.item
        self._info("Adding a new " + model_name + " plot...")

        sub_layout = self._create_new_sub_layout(model_name)

        self._sub_layouts.append(sub_layout)

        new_children = row([lay.layout for lay in self._sub_layouts])
        self.layout.children[self._sl0].children[self._sl1] = new_children
        self._update_checkbox_column()

        self._info("New plot DONE")

    def _del_sub_layout(self, attr, old, new):
        self._info("Deleting layout...")
        removed_i = new[0]
        del self._sub_layouts[removed_i]

        new_children = row([lay.layout for lay in self._sub_layouts])
        self.layout.children[self._sl0].children[self._sl1] = new_children
        self._update_checkbox_column()
        self._info("Deleting DONE")

    def _data_change(self, attr, old, new):
        self._update_immediate_sublayouts()

    def _update_all_sublayouts(self):
        self._info("Updating all sublayouts")

        for sub_lay in self._sub_layouts:
            self._info("Updating figure " + sub_lay.subl_name + "...")
            sub_lay.refit()

        self._info("Updating DONE")

    """Other methods"""

    def _create_data_sandbox(self):
        pass

    def _update_immediate_sublayouts(self):
        sub_layouts_to_update = [sub_lay for sub_lay in self._sub_layouts if sub_lay.immediate_update()]
        if len(sub_layouts_to_update) == 0:
            return

        self._info("Immediate update of sublayouts...")

        for sub_lay in sub_layouts_to_update:
            self._info("Updating figure " + sub_lay.subl_name + "...")
            sub_lay.refit()

        self._info("Updating sublayouts DONE")

    def _update_checkbox_column(self):
        """Updates column with checkbox and dropdown buttons.
        Change labels in checkbox by creating a new one."""
        labels = [sub_lay.subl_name for sub_lay in self._sub_layouts]
        checkbox_button = CheckboxButtonGroup(labels=labels, default_size=200*len(self._sub_layouts),
                                              active=[])
        checkbox_button.on_change('active', self._del_sub_layout)
        self.layout.children[self._dp0].children[self._dp1].children[self._dp2] = \
            row(checkbox_button, self._dropdown)

    @staticmethod
    def _info(message):
        print("Top layout: " + message)

    def _create_new_sub_layout(self, model_name):
        return None


class RegressionGeneralLayout(GeneralLayout):
    def __init__(self, source_data, log=True):
        GeneralLayout.__init__(self, source_data, log)

    """Methods for layout initialisation"""

    def _layout_init(self):
        model_selection = self._model_selection_init()
        fit_all = self._fit_all_init()
        data_sandbox_button = self._sand_box_button_init()

        general_info = Div(
            text="Add a new figure by selecting one through clicking on \"+ add model\". "
                 "Figure can be deleted by clicking on it in the button group. "
                 "You can add a new/move/delete points in the dataset by selecting \"Point Draw Tool\" "
                 "in the figure toolbar. Then choose click anywhere into the figure. You can select "
                 "more points by holding \"shift\" while selecting and delete them by with \"backspace\". "
                 "For more manipulating options with dataset click on \"Data Sandbox\". You can deactivate it by "
                 "clicking on the option again. Every figure can be updated immediately after adding/moving/deleting"
                 " points by selecting \"Immediate update\" bellow the figure (this comes handy with fast algorithms "
                 "as SVM or Bayes).",
            width=1000
        )

        self.layout = column(column(general_info,
                                    row(data_sandbox_button, fit_all, model_selection)
                                    ),
                             row(row(),  # this is a place for data sandbox
                                 row()))  # this is a place for added SubLayouts
        self._sb1, self._sb2 = 1, 0  # data sandbox position
        self._sl0, self._sl1 = 1, 1  # SubLayouts position
        self._dp0, self._dp1, self._dp2 = 0, 1, 2  # DropDown (model_selection) position

    """Methods for interactive calling (called by on_change or on_click triggers)"""

    # def _data_sandbox_trigger(self, attr, old, new):
    #     if 0 in new:  # sandbox button was activated
    #         sandbox = sr.regression_data_sandbox(name="Data Sandbox", source_data=self.source_data)
    #         self.layout.children[self._sb1].children[self._sb2] = sandbox.layout
    #     else:
    #         self.layout.children[self._sb1].children[self._sb2] = row()

    def _data_change(self, attr, old, new):
        self._update_immediate_sublayouts()

    """Other methods"""

    def _create_data_sandbox(self):
        return sr.regression_data_sandbox(name="Data Sandbox", source_data=self.source_data)

    @staticmethod
    def _menu_init():
        """Create options for model selection menu"""
        return list(zip(REG_MODELS.keys(), REG_MODELS.keys()))

    def _create_new_sub_layout(self, model_name):
        """Returns new sublayout with regression figure"""
        return sr.reg_resolution(model_name=model_name,
                                 source_data=self.source_data)


class ClassifierGeneralLayout(GeneralLayout):
    def __init__(self, source_data, log=True):
        GeneralLayout.__init__(self, source_data, log)

    """Methods for layout initialisation"""

    def _layout_init(self):
        model_selection = self._model_selection_init()
        fit_all = self._fit_all_init()
        class_selection = self._class_selection_init()
        data_sandbox_button = self._sand_box_button_init()

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
        self._sb1, self._sb2 = 1, 0  # data sandbox position
        self._sl0, self._sl1 = 1, 1  # SubLayouts position
        self._dp0, self._dp1, self._dp2 = 0, 2, 2  # DropDown (model_selection) position
        self._cs0, self._cs1, self._cs2 = 0, 1, 0  # _class_selection position

    @staticmethod
    def _menu_init():
        """Create options for model selection menu"""
        return list(zip(CLS_MODELS.keys(), CLS_MODELS.keys()))

    def _create_new_sub_layout(self, model_name):
        return sr.cls_resolution(model_name=model_name,
                                 source_data=self.source_data)

    def _class_selection_init(self):
        classes_count = len(self.source_data.uniq_values())
        try:
            old_active = self._class_select_button.active
        except AttributeError:
            old_active = 0

        button_width = CLASS_SELECT_BUTTON_WIDTH * classes_count
        picker_width = CLASS_SELECT_BUTTON_WIDTH - self._normalise_picker_width(classes_count)
        self._color_pickers = [ColorPicker(
            title="",
            width=picker_width, width_policy="fixed",
            color=self.source_data.palette[i]
        ) for i in range(classes_count)]
        for picker in self._color_pickers:
            picker.on_change('color', self._change_color)
        self._class_select_button = RadioButtonGroup(
            labels=self._normalise_uniq_values(self.source_data.uniq_values()),
            active=old_active,
            width=button_width, width_policy="fixed"
        )

        if classes_count > 9:
            return column(row(self._color_pickers),
                          row(self._class_select_button))

        self._new_class_button = Button(label="+ new class", button_type="primary",
                                        width=CLASS_SELECT_BUTTON_WIDTH, width_policy="fixed"
                                        )
        self._new_class_button.on_click(self._new_class)
        return column(row(self._color_pickers),
                      row(self._class_select_button, self._new_class_button))

    """Methods for interactive calling (called by on_change or on_click triggers)"""
    #
    # def _data_sandbox_trigger(self, attr, old, new):
    #     if 0 in new:  # sandbox button was activated
    #         sandbox = sr.classifier_data_sandbox(name="Data Sandbox", source_data=self.source_data,
    #                                              class_select_button=self._class_select_button)
    #         self.layout.children[self._sb1].children[self._sb2] = sandbox.layout
    #     else:
    #         self.layout.children[self._sb1].children[self._sb2] = row()

    def _new_class(self):
        self.source_data.add_new_class(
            class_name=self._new_name(prev=self.source_data.uniq_values()[-1])
        )
        self._data_sandbox.update_slider_classes_count()
        self._class_selection_update()

    def _change_color(self, attr, old, new):

        if new == "#000000":  # two colors check
            for picker in self._color_pickers:
                if picker.color == "#000000":
                    picker.color = old
                    return

        self.source_data.replace_color(
            old_color=old, new_color=new
        )

        for sub_lay in self._sub_layouts:
            sub_lay.update_renderer_colors()

    def _data_change(self, attr, old, new):
        self._info("Updating data...")

        if len(old[self.source_data.x]) < len(new[self.source_data.x]):

            if new['color'][-1] == EMPTY_VALUE_COLOR:
                new_class = self.source_data.uniq_values()[self._class_select_button.active]
                self.source_data.update_cls_newly_added(new_class,
                                                        new_i=len(old[self.source_data.x]))

        self._info("Updating data DONE")

        self._update_immediate_sublayouts()

    """Other methods"""

    def _create_data_sandbox(self):
        return sr.classifier_data_sandbox(name="Data Sandbox", source_data=self.source_data,
                                          class_select_button=self._class_select_button)

    def _class_selection_update(self):
        classes_count = len(self.source_data.uniq_values())
        picker_width = CLASS_SELECT_BUTTON_WIDTH - self._normalise_picker_width(classes_count)
        button_width = CLASS_SELECT_BUTTON_WIDTH * classes_count

        self._color_pickers.append(
            ColorPicker(
                title="",
                width=picker_width, width_policy="fixed",
                color=self.source_data.palette[classes_count - 1]
            )
        )
        self._color_pickers[-1].on_change('color', self._change_color)
        self.layout.children[self._cs0].children[self._cs1].children[self._cs2].children[0] \
            = row(self._color_pickers)

        new_labels = self._normalise_uniq_values(self.source_data.uniq_values())
        self._class_select_button.update(labels=new_labels,
                                         width=button_width)

    def _new_name(self, prev):
        if prev.isdigit():
            return str(int(prev) + 1)
        return "extra " + str(len(self.source_data.uniq_values()))

    @staticmethod
    def _normalise_uniq_values(uniq_values):
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

    @staticmethod
    def _normalise_picker_width(classes_count):
        if classes_count <= 2:
            return 5
        if classes_count <= 4:
            return 7
        if classes_count <= 6:
            return 8
        return 9

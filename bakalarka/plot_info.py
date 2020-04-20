
from bokeh.models import CategoricalColorMapper, ColumnDataSource

from numpy import empty

# TODO: udelat z tohoto backend?
# TODO: pamatovat algoritmy podle id figury


class PlotInfo:
    def __init__(self, df,
                 pol_min_degree, pol_max_degree,
                 palette, x_extension, y_extension):
        self.pol_min_degree = pol_min_degree
        self.pol_max_degree = pol_max_degree
        self.palette = palette
        self.x_extension = x_extension
        self.y_extension = y_extension
        self.immediate_update = False

        uniq_values = sorted(list(set(df['classification'])))
        self.color_mapper = CategoricalColorMapper(palette=self.palette, factors=uniq_values)
        self.color_dict = self.__uniq_vals2color_dict(uniq_values)

        self.plot_source = ColumnDataSource(
            data={
                'x': df['x'].tolist(),
                'y': df['y'].tolist(),
                'classification': df['classification'].tolist()
            }
        )
        # self.plot_source.remove('index')
        self.plot_source.add([self.color_dict[val] for val in df['classification']], 'color')

        self.plot_source_trigger = None

    def set_plot_source_trigger(self, f):
        self.plot_source.on_change('data', f)
        self.plot_source_trigger = f

    def replace_data(self, x, y, classification):
        uniq_values = self.uniq_values()
        for cls in classification:
            if cls not in uniq_values:
                uniq_values.append(cls)
        self.color_mapper = CategoricalColorMapper(palette=self.palette, factors=uniq_values)
        self.color_dict = self.__uniq_vals2color_dict(uniq_values)

        self.plot_source.remove_on_change('data', self.plot_source_trigger)
        self.plot_source.update(
            data=dict(
                x=x.tolist(),
                y=y.tolist(),
                classification=classification,
                color=[self.color_dict[val] for val in classification]
            )
        )
        self.plot_source.on_change('data', self.plot_source_trigger)

        self.append_data(empty(shape=0), empty(shape=0), [])

    def append_data(self, x_new, y_new, classification_new):
        colors = [self.color_dict[cls] for cls in classification_new]
        new_data = {
            'x': x_new.tolist(),
            'y': y_new.tolist(),
            'classification': classification_new,
            'color': colors
        }
        self.plot_source.stream(new_data)

    def uniq_values(self):
        return sorted(self.color_dict.keys())

    def add_new_color(self, class_name):
        """add a new color possibility"""
        prev_last_index = len(self.uniq_values()) - 1
        self.color_dict[class_name] = self.palette[prev_last_index + 1]
        self.color_mapper = CategoricalColorMapper(palette=self.palette, factors=self.uniq_values())

    def update_color_newly_added(self, new_class, new_i):
        """updates color in newly added data
        new_i is starting index of the new rows (where attribute 'color' is 'added')
        """
        self.plot_source.remove_on_change('data', self.plot_source_trigger)  # unsubscribe f, so there wont be an unwanted trigger
        source_len = len(self.plot_source.data['x'])

        patches = {
            'classification': [(i, new_class) for i in range(new_i, source_len)],
            'color': [(i, self.color_dict[new_class]) for i in range(new_i, source_len)]
        }
        self.plot_source.patch(patches)

        self.plot_source.on_change('data', self.plot_source_trigger)  # subscribe again

    def replace_color(self, old_color, new_color):
        self.plot_source.remove_on_change('data', self.plot_source_trigger)
        patch = []
        for i, color in zip(range(len(self.plot_source.data['color'])), self.plot_source.data['color']):
            if color == old_color:  # find old color occurrences and add then into the patch
                patch.append((i, new_color))

        patches = {
            'color': patch
        }
        self.plot_source.patch(patches)

        for cls, color in self.color_dict.items():
            if color == old_color:
                self.color_dict[cls] = new_color

        tmp_palette = list(self.palette)
        for i in range(len(tmp_palette)):
            if tmp_palette[i] == old_color:
                tmp_palette[i] = new_color
        self.palette = tuple(tmp_palette)

        self.color_mapper.palette = self.palette

        self.plot_source.on_change('data', self.plot_source_trigger)

    def __uniq_vals2color_dict(self, uniq_values):
        """Map classification values to integers such as
        ["setosa", "virginica", "setosa"] -> {0: 'setosa', 1: 'virginica'}
        """
        values_dict = {}
        for uq, i in zip(uniq_values, range(len(uniq_values))):
            values_dict[uq] = self.palette[i]
        return values_dict

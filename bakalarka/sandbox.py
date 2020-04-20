from bokeh.layouts import column
from bokeh.models import PointDrawTool, ColumnDataSource, Button, PanTool
from bokeh.plotting import figure, curdoc


source_red = ColumnDataSource(data=dict(
    x=[1, 2, 3, 4, 5],
    y=[2, 5, 8, 2, 7]
))

source_blue = ColumnDataSource(data=dict(
    x=[0, 6, 7, 2, 4],
    y=[2, 5, 8, 2, 7]
))


################################################
data_store = {0: source_red, 1: source_blue}
class_colors = {0: "red", 1: "blue"}
#class_point_draw_icons = {0: "iconred.jpg", 1: "iconblue.jpg"}

fig = figure(x_range=(0, 100), y_range=(0, 100))

class_renderers = {}
class_point_draw_tools = {}

move_circle = fig.circle('x', 'y', color="red", source=source_red, size=7)
point_draw_tool = PointDrawTool(renderers=[move_circle])
fig.add_tools(point_draw_tool)

pan_tool = PanTool()
fig.add_tools(pan_tool)

def activate_first():
    pass


def activate_second():
    fig.toolbar.active_scroll = None
    fig.toolbar.active_drag = pan_tool


first = Button(label="activate first class")
first.on_click(activate_first)

second = Button(label="activate second class")
second.on_click(activate_second)

def toolbar_print(attr, old ,new):
    print(old)
    print(new)
    print("i am walking hereee")

fig.toolbar.on_change('active_drag', toolbar_print)

fig.toolbar.on_change('active_multi', toolbar_print)

lay = column(first, second, fig)
curdoc().add_root(lay)

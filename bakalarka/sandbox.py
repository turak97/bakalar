from bokeh.layouts import row, column
from bokeh.models import PointDrawTool, ColumnDataSource, Button, PanTool, Slider
from bokeh.plotting import figure, curdoc
from bokeh.models import CheckboxButtonGroup

from random import uniform, randint

SOURCES = 4

y = [500] * 365
x = [i for i in range(1, 365 + 1)]
for i in range(1, len(x)):
    y[i] = y[i - 1] + uniform(-20, 21)

sources = []

for i in range(SOURCES):
    y = [randint(450, 750)] * 365
    for j in range(1, len(y)):
        y[j] = y[j - 1] + uniform(-20, 21)
        if y[j] < 0:
            y[j] = 0
    source = ColumnDataSource(
        data=dict(
            x=[i for i in range(1, 365 + 1)],
            y=y
        )
    )
    sources.append(source)

fig = figure()
fig_static = figure(toolbar_location=None)

names = ["ABX", "CBAA", "PLOOD", "LOPX"]

for source, color, name in zip(sources, ["red", "purple", "green", "grey"], names):
    fig.line(x='x', y='y', source=source, legend_label=name, color=color)
    fig_static.line(x='x', y='y', source=source, legend_label=name, color=color)


def button_trigger(attr, old, new):
    for i in range(SOURCES):
        if i in new:
            fig.renderers[i].visible = True
        else:
            fig.renderers[i].visible = False


button = CheckboxButtonGroup(labels=names, active=[i for i in range(SOURCES)])
button.on_change('active', button_trigger)

slider = Slider(start=1, end=100, step=1, value=1)

lay = row(column(fig, button), fig_static, slider)
curdoc().add_root(lay)

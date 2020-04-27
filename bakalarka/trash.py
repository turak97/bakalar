

from bokeh.plotting import figure, show
from bokeh.models import Title, Toolbar, Button
from bokeh.layouts import row

fig = figure()


# fig.toolbar = Toolbar(location=None)


def change():
    button.label = "ccc"


button = Button(label="bla")
button.on_click(change)

lay = row(button, fig)

show(lay)

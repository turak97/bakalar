
from bokeh.io import curdoc

from bokeh.models import Div, Button
from bokeh.layouts import row, column


button = Button(label="bla")

div = Div(text="bla")


def b():
    div.text = "holahou"


button.on_click(b)

col = column(button, div)

curdoc().add_root(col)


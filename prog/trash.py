

from bokeh.io import output_file, show
from bokeh.models import Dropdown
from bokeh.io import curdoc

output_file("dropdown.html")

menu = [("Item 1", "item_1"), ("Item 2", "item_2"), None, ("Item 3", "item_3")]
dropdown = Dropdown(label="Dropdown button", button_type="warning", menu=menu)

def bla(value):
    print(value.item)

dropdown.on_click(bla)

curdoc().add_root(dropdown)


from random import random
from bokeh.layouts import column
from bokeh.models import Button, ColorBar, BasicTicker
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, curdoc
from bokeh.transform import linear_cmap
import bokeh

# create a plot and style its properties
p = figure(x_range=(0, 100), y_range=(0, 100), toolbar_location=None)
p.border_fill_color = 'black'
p.background_fill_color = 'black'
p.outline_line_color = None
p.grid.grid_line_color = None

i = 0

color_mapper = linear_cmap(field_name='text', palette=bokeh.palettes.Cividis[5], low=0, high=i)
color_bar = ColorBar(
    color_mapper=color_mapper['transform'], 
    ticker=BasicTicker(),
    label_standoff=12,
    border_line_color='black',
    location=(0, 0),
    orientation="horizontal")

# add a text renderer to our plot (no data yet)
ds = ColumnDataSource(dict(x=[], y=[], text=[]))
r = p.text(x='x', y='y', text='text', text_color=color_mapper, text_font_size="20pt",
           text_baseline="middle", text_align="center", source=ds)
p.add_layout(color_bar, 'below')

# create a callback that will add a number in a random location
def callback():
    global i

    i = i + 1 
    # BEST PRACTICE --- update .data in one step with a new dict
    new_data = dict()
    new_data['x'] = ds.data['x'] + [random()*70 + 15]
    new_data['y'] = ds.data['y'] + [random()*70 + 15]
    new_data['text'] = ds.data['text'] + [i]
    ds.data = new_data
    color_mapper['transform'].high = i

# add a button widget and configure with the call back
button = Button(label="Press Me")
button.on_click(callback)
#show(p)
# put the button and plot in a layout and add to the document
curdoc().add_root(column(button, p))
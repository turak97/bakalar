import pandas as pd
from bokeh.io import show
from bokeh.models import ColumnDataSource as cds, GMapOptions, TableColumn, DataTable
from bokeh.plotting import gmap
from bokeh.layouts import column,layout
from bokeh.models.glyphs import Circle
from bokeh.plotting import figure

from bokeh.models import ColumnDataSource, Column
from bokeh.plotting import Figure, show
from bokeh.models import PointDrawTool
from bokeh.io import curdoc
import data_gen as dg

# For GMaps to function, Google requires you obtain and enable an API key:
#
#     https://developers.google.com/maps/documentation/javascript/get-api-key
#
# Replace the value below with your personal API key:
p = figure(title='Double click to leave a dot.',
           width=700, height=700,
           x_range=(-100, 100), y_range=(-100, 100))

[x, y] = dg.polynom_data(interval=(-100, 100), clusters=1)


# source = ColumnDataSource(
#     data=dict(lat=[20, -40, 80],
#               lon=[-40, 10, 90])
# )

x, y = x.tolist(), y.tolist()

source = ColumnDataSource(
    data=dict(lat=x,
              lon=y)
)

print(source.data)


c1 = p.circle(x="lon", y="lat", size=15, source=source)

# columns = [TableColumn(field='lon', title="x"),
#            TableColumn(field='lat', title="y"),
#            TableColumn(field='name', title='Name')]

# table = DataTable(source=source, editable=True, height=200) #  , columns=columns

draw_tool = PointDrawTool(renderers=[c1], empty_value='added', add=True)
p.add_tools(draw_tool)
p.toolbar.active_tap = draw_tool

# col = column(p, table)

layout = Column(p)

curdoc().add_root(layout)

from bokeh.plotting import figure, show, ColumnDataSource

line_data = {
    'x': [0, 1, 2, 3, 4],
    'y': [2.5, -1, 4, 2]
}  # definice dat pomoci slovniku
line_source = ColumnDataSource(data=line_data)  # vlozeni slovniku do zdroje
circle_source = ColumnDataSource(
    data=dict(
        x=[1, 2, -1, 1.3],
        y=[3, -2, 0, 0]
    )
)  # vytvoreni druheho zdroje

fig = figure(title="First figure", tools="pan,wheel_zoom,reset")  # vytvoreni figury
fig.line(x='x', y='y', source=line_source, color='purple')  # pridani spojnicoveho grafu
fig.circle(x='x', y='y', source=circle_source, color='pink', size=12)  # pridani bodoveho grafu

show(fig)  # vykresleni figury v prohlizeci

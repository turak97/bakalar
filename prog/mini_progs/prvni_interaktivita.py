from bokeh.layouts import row
from bokeh.plotting import figure, ColumnDataSource, curdoc
from bokeh.models import MultiSelect, CategoricalColorMapper
from bokeh.palettes import Category10_10  # import palety
from bokeh.sampledata.iris import flowers  # import dat, flowers je Pandas dataframe

p = figure(title="Iris Morphology",
           x_axis_label='Petal Length',
           y_axis_label='Petal Width')  # Vytvoreni figury

flowers_source = ColumnDataSource(flowers)  # vytvoreni zdroje pro bodovy renderer (circle)
flowers_species = list(set(flowers['species']))  # ziskani trid v datasetu

# vytvoreni mapovaciho objektu: trida -> barva
color_mapper = CategoricalColorMapper(
    factors=flowers_species, palette=Category10_10)
p.circle(
    source=flowers_source,
    x='petal_length', y='petal_width',
    color={'field': 'species', 'transform': color_mapper},
    fill_alpha=0.2, size=10)  # vytvoreni renderu


def change_selected(attr, old, new):  # funkce, ktera je volana pri zmene MultiSelect widgetu
    new_data = flowers.loc[
        flowers['species'].isin(new)
    ]  # vyber chtenych trid z datasetu
    flowers_source.update(
        data=new_data
    )  # aktualizace zdroje


# vytvoreni widgetu, pomoci ktereho lze zvolit zobrazovane tridy
select_button = MultiSelect(
    options=[(specie, specie) for specie in flowers_species])
# funkce change_selected se bude volat pri kazde zmene atributu 'value'
select_button.on_change('value', change_selected)

layout = row(p, select_button)  # row slouzi pro organizaci objektu do radku
curdoc().add_root(layout)  # create a blank document and add layout objects to it

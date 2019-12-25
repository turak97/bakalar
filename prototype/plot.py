import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import data_gen as dg

import operator

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def lin_plot():

    x_vals, y_vals = dg.generate_lin_data(p=2, disp=10, interval=(0, 100), step=1)

    reg = dg.lin_2D_regression(x_vals, y_vals)

    x_lin, y_lin = dg.lin_coef(reg.coef_[0][0], reg.intercept_[0], x_vals)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers'))
    fig.add_trace(go.Scatter(x=x_lin, y=y_lin, mode='lines'))

    fig.show()


def poly_plot(x, y, min_degree=1, max_degree=10, x_ext=0.3, y_ext=0.3):
    domain_extension = (x[-1] - x[0]) * x_ext
    y_min, y_max = np.min(y), np.max(y)
    range_extension = (y_max - y_min) * y_ext

    fig = go.Figure(
        layout=go.Layout(
            yaxis=dict(autorange=False,
                       range=[y_min - range_extension, y_max + range_extension]),
        )
    )

    # create model and add to the figure line trace of the model
    for degree in np.arange(min_degree, max_degree + 1):
        model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                          ('linear', LinearRegression(fit_intercept=False))])
        # create and fit model and generate data for line
        model = model.fit(x[:, np.newaxis], y)
        model_coef = model.named_steps['linear'].coef_[::-1]  # numpy poly1d expects coefficients in reverse order
        # how polynom really looks like for figure
        f = np.poly1d(model_coef)
        x_from = x[0] - domain_extension
        x_to = x[-1] + domain_extension
        x_plot = np.linspace(x_from, x_to, 1000)
        y_plot = f(x_plot)
        # add line to figure
        fig.add_trace(go.Scatter(
            visible=False,
            x=x_plot, y=y_plot, mode='lines'))

    # create steps for slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)],
            label="Degree " + str(i + min_degree)
        )
        step["args"][1][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    for i in range(degree):
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))

    fig.data[0].visible = True

    # directly create sliders and updating layout
    fig.update_layout(
        sliders=[dict(active=0, steps=steps)]
    )

    fig.show()


data = dg.polynom_data()
poly_plot(data[0], data[1], y_ext=1, x_ext=1)


import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook'
import numpy as np

def iplot(x, y, mode='lines+markers', title='', xlabel='', ylabel=''):
    # https://plot.ly/python/figure-labels/
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(x), y=np.array(y), mode=mode))

    fig.update_layout(
        title=go.layout.Title(
            text=title,
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(text=xlabel)
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(text=ylabel)
        )
    )

    #fig.show()
    return fig

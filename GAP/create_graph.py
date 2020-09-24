#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 下午4:09
# @Author  : Yingying Li

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from igraph import Graph, EdgeSeq
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Create graph matrix')
parser.add_argument('--port', default='2345', type=int, help='vis port')
parser.add_argument('--nodes_number', default='10', type=int, help='nodes number')
args = parser.parse_args()

# initialize app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FBFF'
}
# nodes
# np.random.seed(1)
x = np.random.rand(args.nodes_number)
y = np.random.rand(args.nodes_number)
fig_data = go.Scatter(x=x, y=y, mode='markers')


v_label = list(map(str, range(args.nodes_number)))
G = Graph(1)
G.add_vertex(args.nodes_number)
lay = G.layout('rt')
position = {k: [xi, yi] for k, xi, yi in zip(range(args.nodes_number), x, y)}
labels = v_label

# table
table_size = args.nodes_number


## header
header = html.Div([
    html.Div([
        html.H1(children='Create Graph', style=dict(textAlign='center', color=colors['text'])),
    ])
])

## Body
Body = html.Div([
    html.Div([
        dcc.Markdown((
            """
+ 0) Please choose Canvas Size.
+ 1) Getting the node by your mouse.
+ 2) Connect the corresponding points with the mouse.
            """
        ))
    ]),


    html.Div([
        html.Label('Mouse_model'),
        dcc.RadioItems(
            id='Mouse_model',
            options=[{'label': i, 'value': i} for i in ['Line']],
            value='Line'),
        html.Button('Clear_Graph', id='clear_graph', n_clicks=0),
    ], style=dict(width='50%', display='inline-block')),

    html.Div([
        html.Div([
            html.Button('Save', id='save_table', n_clicks=0),
        ]),
    ], style=dict(width='10%', margin_right='10%', display='inline-block')),

    html.Div([
        dcc.Graph(id='Figure')
    ], style=dict(width='45%', display='inline-block')),

    html.Div([
        dcc.Graph(id='Graph')
    ], style=dict(width='45%', display='inline-block')),

    html.Div([
        dcc.Graph(id='Table'),
    ], style=dict(width='45%', display='inline-block')),
])

app.layout = html.Div([
    header,
    Body,
])
def udate_table(table, idxlist, matrix):

    matrix[idxlist[0], idxlist[1]] = 1
    matrix[idxlist[1], idxlist[0]] = 1
    table.add_trace(go.Table(
        header=dict(
            values=[i for i in labels],
            line_color='darkslategray',
            fill_color='royalblue',
            align='center',
            font=dict(color='white', size=12),
            height=40
        ),
        cells=dict(
            values=[[i for i in labels], matrix],
            line_color='darkslategray',
            fill=dict(color=['royalblue', 'white']),
            align='center',
            font_size=12,
            height=40
        )
    ))
    table.update_layout(title='Table Layout',
                        width=800,
                        height=800
                        )
    return table


def update_graph(G, graph_fig, idxlist, posx, posy, Xe, Ye):

    G.add_edges((idxlist[0], idxlist[1]))
    Xe += [posx[0], posx[1], None]
    Ye += [posy[0], posy[1], None]
    # labels = v_label
    graph_fig.add_trace(go.Scatter(x=Xe,
                               y=Ye,
                               mode='lines',
                               line=dict(color='rgb(210,210,210)', width=1),
                               hoverinfo='none'
                               ))
    graph_fig.add_trace(go.Scatter(x=Xe,
                               y=Ye,
                               mode='markers',
                               name='bla',
                               marker=dict(symbol='circle-dot',
                                           size=18,
                                           color='#6175c1',  # '#DB4551',
                                           line=dict(color='rgb(50,50,50)', width=1)
                                           ),
                               text=labels,
                               hoverinfo='text',
                               opacity=0.8
                               ))

    def make_annotations(pos, text, font_size=10, font_color='rgb(250,250,250)'):
        L = len(pos)
        if len(text) != L:
            raise ValueError('The lists pos and text must have the same len')
        annotations = []
        for k in range(L):
            annotations.append(
                dict(
                    text=labels[k],  # or replace labels with a different list for the text within the circle
                    x=pos[k][0], y=pos[k][1],
                    xref='x1', yref='y1',
                    font=dict(color=font_color, size=font_size),
                    showarrow=False)
            )
        return annotations

    axis = dict(showline=False,  # hide axis line, grid, ticklabels and  title
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                )

    graph_fig.update_layout(title='Graph Layout',
                      annotations=make_annotations(position, v_label),
                      font_size=12,
                      showlegend=False,
                      xaxis=axis,
                      yaxis=axis,
                      margin=dict(l=40, r=40, b=85, t=100),
                      hovermode='closest',
                      plot_bgcolor='rgb(248,248,248)'
                      )
    return graph_fig


@app.callback(
    [dash.dependencies.Output('Figure', 'figure'),
     dash.dependencies.Output('Graph', 'figure'),
     dash.dependencies.Output('Table', 'figure'),
     ],
    [dash.dependencies.Input('Mouse_model', 'value'),
     dash.dependencies.Input('save_table', 'n_clicks'),
     dash.dependencies.Input('Figure', 'clickData'),
     dash.dependencies.Input('clear_graph', 'n_clicks'),
    ]
)
def update_create_canvas(mouse_model, save_click, figure_click, clear_click):
    fig = go.FigureWidget([fig_data])
    matrix = np.zeros((table_size, table_size))
    maxpoints = 2
    idxlist = []
    Xe = []
    Ye = []
    posx = []
    posy = []

    if mouse_model == "Line":
        if figure_click is None:
            graph_fig = go.Figure(go.Scatter(x=[0, 0], y=[0, 0]))
            tab_fig = go.Figure(go.Scatter(x=[0, 0], y=[0, 0]))
        else:
            tab_fig = go.Figure()
            graph_fig = go.Figure()

            maxpoints -= 1
            print("maxpoints", maxpoints)
            idx = int(figure_click['points'][0]['pointIndex'])
            xi = figure_click['points'][0]['x']
            yi = figure_click['points'][0]['y']
            idxlist.append(idx)
            print("idxlist:", idxlist)
            posx.append(xi)
            posy.append(yi)
            if maxpoints == 0:
                # updated table
                tab_fig = udate_table(tab_fig, idxlist, matrix)

                # updated graph
                graph_fig = update_graph(G, graph_fig, idxlist, posx, posy, Xe, Ye)

                # clear points
                maxpoints = 2
                idxlist = []
                Xe = []
                Ye = []
                posx = []
                posy = []





    if save_click:
        ##save np
        pass
    if clear_click:
        fig = go.Figure(go.Scatter(x=[0, 0], y=[0, 0]))
        graph_fig = go.Figure(go.Scatter(x=[0, 0], y=[0, 0]))
        tab_fig = go.Figure(go.Table())

    # return [f"Choose {mouse_model}", f, matrix]
    return [fig, graph_fig, tab_fig]

if __name__ == "__main__":
    app.run_server(debug=True)
    # app.run_server(host='localhost', port=args.port, debug=False)


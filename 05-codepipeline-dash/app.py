import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

import pandas as pd
from datetime import date


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# DATA
df = pd.read_parquet('/tmp/social_network.parquet')

# LAYOUT
app.layout = html.Div([
    html.H1('Dashboard Social Networks', style={
            "text-align": "center", "margin-top": "24px", "margin-bottom": "48px"}),
    html.Div([
        html.Label('Datetime Range'),
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=date(2021, 1, 1),
            end_date=date(2021, 4, 30),
        ),
        html.Label('Social Networks'),
        dcc.Dropdown(
            id="social-networks-dropdown",
            options=[{"label": social_network, "value": social_network}
                     for social_network in df.social_network.unique()],
            value=[social_network for social_network in df.social_network.unique()],
            multi=True
        ),
        html.Label('Devices'),
        dcc.Checklist(
            id='devices-checkbox',
            options=[{"label": device, "value": device}
                     for device in df.device.unique()],
            value=[device for device in df.device.unique()],
            labelStyle={'display': 'inline-block'}
        )
    ], style={"columnCount": 3, 'textAlign': "center", "margin-top": "24px", "margin-bottom": "48px"}),
    html.Div([
        html.Div([
            html.Img(src="https://upload.wikimedia.org/wikipedia/commons/9/99/Sample_User_Icon.png",
                     style={"width": "50px"}),
            html.H2(
                id='total-visit',
            )
        ]),
        html.Div([
            html.Img(src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Facebook_logo_%28square%29.png/240px-Facebook_logo_%28square%29.png",
                     style={"width": "50px"}),
            html.H2(
                id='facebook-visit',
            )
        ]),
        html.Div([
            html.Img(src="data:image/jpeg;base64,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",
                     style={"width": "50px"}),
            html.H2(
                id='instagram-visit',
            )
        ]),
        html.Div([
            html.Img(src="https://logos-marcas.com/wp-content/uploads/2020/04/Twitter-Logo.png",
                     style={"width": "100px"}),
            html.H2(
                id='twitter-visit',
            )
        ]),
    ], style={"columnCount": 4, 'textAlign': "center"}),
    html.H3('Total Visits by Month', style={"textAlign": "center"}),
    dcc.Graph(
        id='total-visit-line'
    ),
    html.H3('Total Visits by Social Networks', style={"textAlign": "center"}),
    dcc.Graph(
        id='total-visit-social-networks-line'
    ),
    html.Div([
        html.H3('Total Visits by Country', style={"textAlign": "center"}),
        dcc.Graph(
            id='world-map'
        ),
        html.H3('Total Visits by Device', style={"textAlign": "center"}),
        dcc.Graph(
            id='diveces-pie'
        )
    ], style={"columnCount": 2})
])


@app.callback(
    Output('total-visit', 'children'),
    Output('facebook-visit', 'children'),
    Output('instagram-visit', 'children'),
    Output('twitter-visit', 'children'),
    Output('total-visit-line', 'figure'),
    Output('total-visit-social-networks-line', 'figure'),
    Output('world-map', 'figure'),
    Output('diveces-pie', 'figure'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
    Input('social-networks-dropdown', 'value'),
    Input('devices-checkbox', 'value'))
def update_figures(start_date_selected, end_date_selected, social_networks_selected, devices_selected):

    total_visit = (
        df
        .loc[(df.social_network.isin(social_networks_selected)) &
             (df.device.isin(devices_selected)) &
             (df.datetime >= start_date_selected) &
             (df.datetime <= end_date_selected)]
    ).shape[0]

    facebook_visit = (
        df
        .loc[(df.social_network == 'facebook') &
             (df.social_network.isin(social_networks_selected)) &
             (df.device.isin(devices_selected)) &
             (df.datetime >= start_date_selected) &
             (df.datetime <= end_date_selected)]
    ).shape[0]

    instagram_visit = (
        df
        .loc[(df.social_network == 'instagram') &
             (df.social_network.isin(social_networks_selected)) &
             (df.device.isin(devices_selected)) &
             (df.datetime >= start_date_selected) &
             (df.datetime <= end_date_selected)]
    ).shape[0]

    twitter_visit = (
        df
        .loc[(df.social_network == 'twitter') &
             (df.social_network.isin(social_networks_selected)) &
             (df.device.isin(devices_selected)) &
             (df.datetime >= start_date_selected) &
             (df.datetime <= end_date_selected)]
    ).shape[0]

    df_by_month = (
        df
        .loc[(df.social_network.isin(social_networks_selected)) &
             (df.device.isin(devices_selected)) &
             (df.datetime >= start_date_selected) &
             (df.datetime <= end_date_selected)]
        .groupby(['year', 'month'])
        .count()
        .name
        .reset_index()
        .assign(
            year_month=lambda df: df.year+'-'+df.month
        )
    )

    df_by_month_social_networks = (
        df
        .loc[(df.social_network.isin(social_networks_selected)) &
             (df.device.isin(devices_selected)) &
             (df.datetime >= start_date_selected) &
             (df.datetime <= end_date_selected)]
        .groupby(['year', 'month', 'social_network'])
        .count()
        .name
        .reset_index()
        .assign(
            year_month=lambda df: df.year+'-'+df.month
        )
    )

    df_country = (
        df
        .loc[(df.social_network.isin(social_networks_selected)) &
             (df.device.isin(devices_selected)) &
             (df.datetime >= start_date_selected) &
             (df.datetime <= end_date_selected)]
        .groupby(['country_code', 'country'])
        .count()
        .name
        .reset_index()
    )

    df_devices = (
        df
        .loc[(df.social_network.isin(social_networks_selected)) &
             (df.device.isin(devices_selected)) &
             (df.datetime >= start_date_selected) &
             (df.datetime <= end_date_selected)]
        .groupby(['device'])
        .count()
        .name
        .reset_index()
    )

    total_visit_fig = px.line(
        df_by_month,
        x="year_month",
        y="name",
        labels={
            "name": "Total Visits",  "year_month": "Month"
        }
    )

    total_visit_social_network_fig = px.line(
        df_by_month_social_networks,
        x="year_month",
        y="name",
        color="social_network",
        labels={
            "name": "Total Visits",  "year_month": "Month"
        }
    )

    world_map_fig = px.choropleth(
        df_country,
        locations='country_code',
        color="name",
        hover_name="country",
        color_continuous_scale='plasma',
        labels={
            'name': 'Total Visits'
        }
    )

    devices_pie_fig = px.pie(
        df_devices,
        values='name',
        names='device',
        labels={
            'name': 'Total Visits'
        }
    )

    return total_visit, facebook_visit, instagram_visit, twitter_visit, total_visit_fig, total_visit_social_network_fig, world_map_fig, devices_pie_fig


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port="80")

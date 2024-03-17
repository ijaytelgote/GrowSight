import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class VisualizationDashboard:
    def __init__(self, data):
        self.data = data
        self.dropdown_options = [
            {"label": col, "value": col} for col in ["Monthly Revenue", "Opportunity Amount", "Support Tickets Open",
                                                     "Support Tickets Closed", "Lead Score", "Age", "Contract Type",
                                                     "Gender", "Lead Status"]
        ]
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)

    def create_scatter_layout(self):
        return html.Div([
            html.H1('Scatter Plot'),
            dcc.Dropdown(
                id='scatter-dropdown-x',
                options=self.dropdown_options,
                value='Monthly Revenue',
                style={'color': 'black'}
            ),
            dcc.Dropdown(
                id='scatter-dropdown-y',
                options=self.dropdown_options,
                value='Opportunity Amount',
                style={'color': 'black'}
            ),
            dcc.Graph(
                id='scatter-plot',
                style={'width': '100%', 'height': '90%'}
            )
        ])
    
    def create_pie_chart_layout(self):
        return html.Div([
            html.H1("World GDP Distribution by Category"),
            dcc.Dropdown(
                id='pie-dropdown-category',
                options=[
                    {'label': 'Age Group', 'value': 'Age'},
                    {'label': 'Lead Status', 'value': 'Lead Status'},
                    {'label': 'Contract Type', 'value': 'Contract Type'},
                    {'label': 'Continent', 'value': 'Continent'},
                    {'label': 'Gender', 'value': 'Gender'},
                ],
                value='Age',
                clearable=False
            ),
            dcc.Dropdown(
                id='pie-dropdown-year',
                options=[{'label': year, 'value': year} for year in range(1980, 2021, 5)],
                value=range(1980, 2021, 5)[-1],
                clearable=False
            ),
            dcc.Graph(id='gdp-pie-chart')
        ])

    def create_time_series_layout(self):
        return html.Div([
            dcc.Dropdown(
                id='time-dropdown-x',
                options=[{'label': col, 'value': col} for col in ['Last Email Sent Date','Last Interaction Date','Last Phone Call Date','Last Meeting Date']],
                value='Last Email Sent Date',
                style={'width': '48%', 'display': 'inline-block'}
            ),
            dcc.Dropdown(
                id='time-dropdown-y',
                options=[{'label': col, 'value': col} for col in ['Monthly Revenue','Opportunity Amount','Probability of Close']],
                value='Opportunity Amount',
                style={'width': '48%', 'float': 'right', 'display': 'inline-block'}
            ),
            dcc.Graph(id='line-chart')
        ])

    def create_bar_chart_layout(self):
        return html.Div([
            html.H1("Fascinating Dashboard", style={'marginBottom': '20px'}),
            html.Div([
                html.Label("Select Column:", style={'marginRight': '10px'}),
                dcc.Dropdown(
                    id='bar-dropdown-column',
                    options=[{'label': col, 'value': col} for col in self.data.columns],
                    value='Gender'
                ),
                dcc.Graph(id='bar-chart'),
                html.Div(id='stats', style={'marginTop': '20px'})
            ], style={'width': '65%', 'margin': 'auto'}),
            html.Div([
                dcc.Slider(
                    id='bar-slider-points',
                    min=10,
                    max=100,
                    step=10,
                    value=50,
                    marks={i: str(i) for i in range(10, 101, 10)},
                    tooltip={'placement': 'top'}
                )
            ], style={'width': '50%', 'margin': 'auto', 'marginTop': '50px'})
        ], style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif'})

    def create_choropleth_layout(self):
        return html.Div([
            html.H1("Country Data Visualization"),
            html.Div([
                dcc.Dropdown(
                    id='choropleth-dropdown-column',
                    options=[
                        {'label': 'Population', 'value': 'Population'},
                        {'label': 'Area (sq km)', 'value': 'Area (sq km)'},
                        {'label': 'GDP (USD)', 'value': 'GDP (USD)'}
                    ],
                    value=['Population'],  # Default value
                    multi=True
                )
            ]),
            html.Div(id='choropleth-container')
        ])

    def create_histogram_layout(self):
        return html.Div([
            html.H1("Fascinating Histogram"),
            html.Div([
                html.Label("Select Data Column:"),
                dcc.Dropdown(
                    id='hist-dropdown-column',
                    options=[{'label': col, 'value': col} for col in self.data.keys()],
                    value='Lead Score'
                ),
                dcc.Graph(id='histogram',
                          config={'displayModeBar': False}),
                html.Div(id='explanation', style={'padding': 10, 'fontSize': 18}),
                html.Label("Number of Bins:"),
                dcc.Slider(
                    id='hist-slider-bins',
                    min=5,
                    max=50,
                    step=1,
                    value=20,
                    marks={i: str(i) for i in range(5, 51, 5)}
                ),
            ], style={'width': '80%', 'margin': 'auto'}),
        ], style={'textAlign': 'center'})

    def run_server(self):
        self.app.run_server(debug=True)

    def update_scatter_plot(self, x_column, y_column):
        fig = px.scatter(self.data, x=x_column, y=y_column, size="Size", color="Continent",
                         log_x=True, size_max=45, title="Scatter Plot")
        fig.update_traces(marker=dict(sizemin=1))  # Set minimum size for markers
        return fig

    def update_pie_chart(self, selected_category, selected_year):
        df_grouped = self.data.groupby(selected_category)['GDP (USD)'].sum()

        # Create a pie chart
        fig = go.Figure(data=[go.Pie(labels=df_grouped.index, values=df_grouped.values, hole=0.4)])

        # Update layout
        title = f"World GDP Distribution by {selected_category.capitalize()}"
        fig.update_layout(title=title,
                          margin=dict(t=50, b=10, l=10, r=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                          uniformtext_minsize=12, uniformtext_mode='hide')

        return fig

    def update_time_series(self, x_value, y_value):
        fig = px.line(self.data, x=x_value, y=y_value, title='Time Series')
        fig.update_xaxes(rangeslider_visible=True)
        return fig

    def update_bar_chart(self, selected_column, num_points):
        counts = self.data[selected_column].value_counts().head(num_points)  # Limit the number of points
        x = counts.index
        y = counts.values

        bar_chart = go.Bar(x=x, y=y, marker=dict(color='royalblue', opacity=0.7))
        layout = go.Layout(title=f'{selected_column} Distribution',
                           xaxis=dict(title=selected_column),
                           yaxis=dict(title='Count'))

        return {'data': [bar_chart], 'layout': layout}

    def update_choropleth(self, selected_columns):
        fig = px.choropleth(
            self.data,
            locations='Country',
            locationmode='country names',
            color=selected_columns[0],  # Take only the first selected column for now
            title='Country Data Visualization',
            color_continuous_scale=px.colors.sequential.Plasma,
            labels={selected_columns[0]: selected_columns[0]},
        )

        if len(selected_columns) > 1:
            for column in selected_columns[1:]:
                fig.add_trace(px.choropleth(
                    self.data,
                    locations='Country',
                    locationmode='country names',
                    color=column,
                    color_continuous_scale=px.colors.sequential.Plasma,
                    labels={column: column}
                ).data[0])

        fig.update_layout(autosize=True)
        return dcc.Graph(figure=fig)

    def update_histogram(self, column, bins):
        x_data = self.data[column]  # Renamed to x_data to avoid conflict with data variable
        histogram_data = [go.Histogram(x=x_data, nbinsx=bins, marker=dict(color='royalblue'))]

        layout = go.Layout(title=f'Histogram of {column}',
                           xaxis=dict(title=column),
                           yaxis=dict(title='Frequency'),
                           bargap=0.05)

        explanation_text = f"The histogram above displays the distribution of {column.lower()} with {bins} bins."

        return {'data': histogram_data, 'layout': layout}, explanation_text

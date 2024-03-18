import os
import random
import tabulate
import dash
import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
from faker import Faker
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import \
    create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI, OpenAI


fake = Faker()

class VisualizationDashboard:
    def __init__(self):
        self.data_generator = DataGenerator()
        self.data = self.data_generator.generate_fake_data()
        self.dropdown_options = [
            {"label": col, "value": col} for col in ["Monthly Revenue", "Opportunity Amount", "Support Tickets Open",
                                                     "Support Tickets Closed", "Lead Score", "Age", "Contract Type",
                                                     "Gender", "Lead Status"]
        ]

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

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
dashboard = VisualizationDashboard()

# Define layout for sentiment analysis page
'''
# Define layout for sentiment analysis page
sentiment_layout = html.Div([
    html.H1("Sentiment Analysis", style={'text-align': 'center', 'margin-bottom': '30px'}),
    html.Div([
        dcc.Textarea(id='input-text', placeholder='Enter text...', style={'width': '60%', 'height': '200px', 'margin': 'auto', 'display': 'block'}),
        html.Button('Analyze', id='analyze-button', n_clicks=0, style={'margin-top': '20px', 'margin-bottom': '20px'}),
        html.Div(id='output-sentiment', style={'text-align': 'center'})
    ], style={'text-align': 'center', 'margin': 'auto', 'margin-top': '50px'}),
    dcc.Graph(id='sentiment-pie-chart', style={'width': '60%', 'margin': 'auto', 'margin-top': '50px', 'display': 'none'})
])
'''

# Store user queries and outputs
user_queries = []

# Define layout
smartdata_layout = html.Div(children=[
    html.H1("SmartDataFrame Chat", style={'textAlign': 'center', 'fontSize': 36, 'marginBottom': 30, 'color': '#333'}),
    dcc.Input(id='user-input', type='text', placeholder='Enter your query...', style={'width': '100%', 'padding': '15px', 'fontSize': '18px', 'marginBottom': '20px', 'borderRadius': '8px', 'border': '1px solid #ccc', 'outline': 'none'}),
    html.Button('Analyse', id='analyse-button', n_clicks=0, style={'backgroundColor': '#4CAF50', 'border': 'none', 'color': 'white', 'padding': '15px 32px', 'textAlign': 'center', 'textDecoration': 'none', 'display': 'inline-block', 'fontSize': '16px', 'marginBottom': '20px', 'cursor': 'pointer', 'borderRadius': '8px'}),
    html.Button('Refresh', id='refresh-button', n_clicks=0, style={'backgroundColor': '#008CBA', 'border': 'none', 'color': 'white', 'padding': '15px 32px', 'textAlign': 'center', 'textDecoration': 'none', 'display': 'inline-block', 'fontSize': '16px', 'marginBottom': '20px', 'marginLeft': '10px', 'cursor': 'pointer', 'borderRadius': '8px'}),
    html.Div(id='output-container', style={'width': '100%', 'padding': '15px', 'fontSize': '16px', 'marginBottom': '20px', 'borderRadius': '8px', 'border': '1px solid #ccc', 'outline': 'none', 'height': '200px', 'overflowY': 'scroll'})
])
# Define layout for default page
default_layout = html.Div([
    dashboard.create_scatter_layout(),
    html.Hr(),
    dashboard.create_pie_chart_layout(),
    html.Hr(),
    dashboard.create_time_series_layout(),
    html.Hr(),
    dashboard.create_bar_chart_layout(),
    html.Hr(),
    dashboard.create_choropleth_layout(),
    html.Hr(),
    dashboard.create_histogram_layout()
])

# Define callback to update layout based on path
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/sentiment':
        return None
    elif pathname == '/talk_to_data':
        return smartdata_layout
    else:
        return default_layout

# Define the main app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

'''# Define function for sentiment analysis using TextBlob
def perform_sentiment_analysis(text):
    analysis = TextBlob(text)
    # Get polarity of the text, which ranges from -1 to 1
    polarity = analysis.sentiment.polarity
    # Classify polarity as positive, negative, or neutral
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Define callback to update sentiment analysis results
@app.callback(
    [Output('output-sentiment', 'children'),
     Output('sentiment-pie-chart', 'figure')],
    [Input('analyze-button', 'n_clicks')],
    [dash.dependencies.State('input-text', 'value')]
)'''
'''
def update_sentiment_analysis(n_clicks, input_text):
    if n_clicks == 0:
        return '', {}
    else:
        sentiment = perform_sentiment_analysis(input_text)
        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        sentiment_counts[sentiment] += 1

        # Create pie chart
        labels = list(sentiment_counts.keys())
        values = list(sentiment_counts.values())

        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig.update_layout(title='Sentiment Distribution')
        
        return f'Sentiment: {sentiment}', fig'''

# Instantiate OpenAI language model

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
     dashboard.data,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

# Define callback to interact with SmartDataFrame
# Define callback to handle user input and display output
@app.callback(
    Output('output-container', 'children'),
    [Input('analyse-button', 'n_clicks'),
     Input('refresh-button', 'n_clicks')],
    [State('user-input', 'value')]
)
def update_output(analyse_clicks, refresh_clicks, user_input):
    global user_queries
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    if triggered_id == 'analyse-button' and analyse_clicks > 0 and user_input:
        user_queries.append({'input': user_input, 'output': agent.run(user_input)})
    elif triggered_id == 'refresh-button' and refresh_clicks > 0:
        user_queries = []
    
    return [html.Div([
        html.P(query['output']) if isinstance(query['output'], str) else dcc.Graph(figure=query['output'])
    ]) for query in user_queries]



# Scatter Plot Callback
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('scatter-dropdown-x', 'value'),
     Input('scatter-dropdown-y', 'value')]
)
def update_scatter_plot(x_column, y_column):
    fig = px.scatter(dashboard.data, x=x_column, y=y_column, size="Size", color="Continent",
                     log_x=True, size_max=45, title="Scatter Plot")
    fig.update_traces(marker=dict(sizemin=1))  # Set minimum size for markers
    return fig

# Pie Chart Callback
@app.callback(
    Output('gdp-pie-chart', 'figure'),
    [Input('pie-dropdown-category', 'value'),
     Input('pie-dropdown-year', 'value')]
)
def update_pie_chart(selected_category, selected_year):
    df_grouped = dashboard.data.groupby(selected_category)['GDP (USD)'].sum()

    # Create a pie chart
    fig = go.Figure(data=[go.Pie(labels=df_grouped.index, values=df_grouped.values, hole=0.4)])

    # Update layout
    title = f"World GDP Distribution by {selected_category.capitalize()}"
    fig.update_layout(title=title,
                      margin=dict(t=50, b=10, l=10, r=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      uniformtext_minsize=12, uniformtext_mode='hide')

    return fig

# Time Series Callback
@app.callback(
    Output('line-chart', 'figure'),
    [Input('time-dropdown-x', 'value'),
     Input('time-dropdown-y', 'value')]
)
def update_graph(x_value, y_value):
    fig = px.line(dashboard.data, x=x_value, y=y_value, title='Time Series')
    fig.update_xaxes(rangeslider_visible=True)
    return fig

# Bar Chart Callback
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('bar-dropdown-column', 'value'),
     Input('bar-slider-points', 'value')]
)
def update_bar_chart(selected_column, num_points):
    counts = dashboard.data[selected_column].value_counts().head(num_points)  # Limit the number of points
    x = counts.index
    y = counts.values

    bar_chart = go.Bar(x=x, y=y, marker=dict(color='royalblue', opacity=0.7))
    layout = go.Layout(title=f'{selected_column} Distribution',
                       xaxis=dict(title=selected_column),
                       yaxis=dict(title='Count'))

    return {'data': [bar_chart], 'layout': layout}

# Choropleth Callback
@app.callback(
    Output('choropleth-container', 'children'),
    [Input('choropleth-dropdown-column', 'value')]
)
def update_choropleth(selected_columns):
    fig = px.choropleth(
        dashboard.data,
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
                dashboard.data,
                locations='Country',
                locationmode='country names',
                color=column,
                color_continuous_scale=px.colors.sequential.Plasma,
                labels={column: column}
            ).data[0])

    fig.update_layout(autosize=True)
    return dcc.Graph(figure=fig)

# Histogram Callback
@app.callback(
    [Output('histogram', 'figure'),
     Output('explanation', 'children')],
    [Input('hist-dropdown-column', 'value'),
     Input('hist-slider-bins', 'value')]
)
def update_histogram(column, bins):
    x_data = dashboard.data[column]  # Renamed to x_data to avoid conflict with data variable
    histogram_data = [go.Histogram(x=x_data, nbinsx=bins, marker=dict(color='royalblue'))]

    layout = go.Layout(title=f'Histogram of {column}',
                       xaxis=dict(title=column),
                       yaxis=dict(title='Frequency'),
                       bargap=0.05)

    explanation_text = f"The histogram above displays the distribution of {column.lower()} with {bins} bins."

    return {'data': histogram_data, 'layout': layout}, explanation_text

if __name__ == '__main__':
    app.run_server(debug=True)

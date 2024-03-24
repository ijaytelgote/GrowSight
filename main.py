

import os
import random
import re
import string
from heapq import nlargest

import dash
import duckdb
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tabulate
from GeneratedData import DataGenerator
from dash import dcc, html
from dash.dependencies import Input, Output, State
from faker import Faker
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import \
    create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI, OpenAI
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob

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
                value='Continent',
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
app = dash.Dash(__name__,suppress_callback_exceptions=True,
                 meta_tags=[{'name': 'viewport',
                             'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}]
                )

dashboard = VisualizationDashboard()


# Function to analyze sentiment using TextBlob
def analyze_sentiment(comment):
    analysis = TextBlob(comment)
    if analysis.sentiment.polarity > 0.2:
        return 'Happy'
    elif analysis.sentiment.polarity < -0.2:
        return 'Sad'
    else:
        return 'Neutral'

# Define the SeriesVisualization class

def customer_seg():
    df = dashboard.data
    X = df[['Monthly Revenue', 'Opportunity Amount', 'Support Tickets Open',
            'Support Tickets Closed', 'Lead Score', 'Age', 'Size',
            'Population', 'Area (sq km)', 'GDP (USD)', 'Probability of Close']]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit KMeans clustering algorithm
    k = 3  # Number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Map clusters to meaningful categories
    cluster_mapping = {0: "Active", 1: "Inactive", 2: "Lead"}

    df['Cluster'] = df['Cluster'].map(cluster_mapping)

    # Visualize in 3D scatter plot
    fig = px.scatter_3d(df, x='Monthly Revenue', y='Opportunity Amount', z='Support Tickets Open',
                         color='Cluster', symbol='Cluster', opacity=0.7,
                         hover_data=['Age', 'Size', 'Population', 'Area (sq km)', 'GDP (USD)', 'Probability of Close'])
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=2, r=5, t=90, b=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )

    return fig

def generate_layout_customer_seg():
    scatter_plot = customer_seg()

    return html.Div([
        html.Div([
            html.H1("Customer Segmentation 3D"),
            html.Div([dcc.Graph(id='scatter-plot', figure=scatter_plot)]),
        ])
    ])


df = dashboard.data

# Adding sentiment analysis to the DataFrame
df['Sentiment'] = df['comment'].apply(analyze_sentiment)

# Creating DataFrame for sentiment analysis
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Sort DataFrame by Timestamp
df.sort_values(by='Timestamp', inplace=True)




sentiment_colors = {
    'Sad': '#1f77b4',  # blue
    'Neutral': '#ff7f0e',  # orange
    'Happy': '#2ca02c',  # green
}



def master_layout():
    return html.Div([
        html.Div([
            html.H1("Sentiment Analysis Over Time"),
            dcc.DatePickerRange(
                id='date-range-picker',
                min_date_allowed=df['Timestamp'].min(),
                max_date_allowed=df['Timestamp'].max(),
                initial_visible_month=df['Timestamp'].min(),
                start_date=df['Timestamp'].min(),
                end_date=df['Timestamp'].max(),
                display_format='YYYY-MM-DD'
            ),
            dcc.Graph(id='sentiment-graph'),
        ]),
        generate_layout_customer_seg(),

        html.Div(
            className='container',
            children=[
                html.Div(
                    className='header',
                    children=[
                        html.H1('Average Customers Fall', className='title'),
                        html.P('Select Date Range:', className='date-label'),
                        dcc.DatePickerRange(
                            id='date-picker',
                            start_date=df['Timestamp'].min(),
                            end_date=df['Timestamp'].max(),
                            display_format='YYYY-MM-DD',
                            className='date-picker'
                        ),
                    ]
                ),
                dcc.Graph(id='sentiment-bubbles', className='graph')]),

        html.Div([
            html.H1("Text Insight Analyzer"),
            html.Div(children='Enter text to analyze:'),
            dcc.Textarea(
                id='input-text',
                value='',
                style={'width': '100%', 'height': 200}
            ),
            html.Label('Select Number of Top Words:'),
            dcc.Dropdown(
                id='top-words-dropdown',
                options=[
                    {'label': 'Top 5', 'value': 5},
                    {'label': 'Top 10', 'value': 10},
                    {'label': 'Top 15', 'value': 15},
                    {'label': 'Top 20', 'value': 20}
                ],
                value=5,
                clearable=False,
                style={'width': '50%'}
            ),
            html.Button('Analyze', id='analyze-button', n_clicks=0, style={'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'border-radius': '5px'}),
            html.Div(id='output-div'),
            dcc.Graph(id='frequency-plot')
        ])])



# Function to scale marker size based on sentiment occurrences
def scale_marker_size(count):
    if count <= 5:
        return 20
    elif count >= 50:
        return 100
    else:
        return count * 2

# Callback to update bubble chart
@app.callback(
    Output('sentiment-bubbles', 'figure'),
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def update_bubble_chart(start_date, end_date):
    filtered_df = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]

    # Count occurrences of each sentiment
    sentiment_counts = filtered_df['Sentiment'].value_counts()

    # Create bubble chart
    data = []
    for sentiment, count in sentiment_counts.items():
        marker_size = scale_marker_size(count)
        data.append(go.Scatter(
            x=[sentiment],
            y=[0],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=sentiment_colors[sentiment],
                line=dict(color='#000000', width=1),  # Add black border
            ),
            name=sentiment,
            hoverinfo='text',
            hovertext=f'Sentiment: {sentiment}<br>Occurrences: {count}',
        ))

    layout = go.Layout(
        title='Sentiment Occurrences',
        xaxis=dict(title='Sentiment', showgrid=False),
        yaxis=dict(visible=False),
        showlegend=False,
        plot_bgcolor='#FFFFFF',  # Set background color
        paper_bgcolor='#f9f9f9',  # Set plot area background color
        margin=dict(l=1, r=1, t=50, b=1)  # Add margin for better display
    )

    return {'data': data, 'layout': layout}

# Define callback to update sentiment graph
@app.callback(
    Output('sentiment-graph', 'figure'),
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_graph(start_date, end_date):
    filtered_df = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]
    
    # Create plot
    trace = go.Scatter(
        x=filtered_df['Timestamp'],
        y=filtered_df['Sentiment'].apply(lambda x: 1 if x == 'Happy' else (-1 if x == 'Sad' else 0)),
        mode='lines+markers',
        name='Sentiment',
        marker=dict(color=['green' if x == 'Happy' else ('red' if x == 'Sad' else 'blue') for x in filtered_df['Sentiment']])
    )

    layout = go.Layout(
        title='Sentiment Analysis Over Time',
        xaxis=dict(title='Timestamp', showgrid=False, zeroline=False),
        yaxis=dict( tickvals=[-1, 0, 1], ticktext=['Sad', 'Neutral', 'Happy'],showgrid=False, zeroline=False),
                       margin=dict(l=1, r=1, t=1, b=40))
    

    return {'data': [trace], 'layout': layout}



# Define callback to update output and frequency plot
@app.callback(
    [Output('output-div', 'children'),
     Output('frequency-plot', 'figure')],
    [Input('analyze-button', 'n_clicks')],
    [dash.dependencies.State('input-text', 'value'),
     dash.dependencies.State('top-words-dropdown', 'value')]
)

           
           
def update_output(n_clicks, text, top_words):
    if not text:
        return "No text to analyze.", {}

    # Tokenize and filter out stop words, punctuation, and non-alphabetic characters
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [re.sub(r'[^a-zA-Z]', '', word) for word in tokens if (word not in stop_words and word not in string.punctuation and re.sub(r'[^a-zA-Z]', '', word))]

    freq_dist = nltk.FreqDist(filtered_tokens)
    insight_result = freq_dist.most_common(top_words)

    # Sentiment analysis
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity

    # Determine sentiment label and color
    sentiment_label = "Happy 😊" if sentiment_score >= 0 else "Sad 😢"
    sentiment_color = '#28a745' if sentiment_score >= 0 else '#dc3545'

    # Generate frequency plot
    df = px.bar(x=[word[0] for word in insight_result], y=[word[1] for word in insight_result], title=f'Top {top_words} Most Common Words')
    df.update_layout(plot_bgcolor='#f9f9f9', paper_bgcolor='#f9f9f9', font_color='#333333',
                      margin=dict(t=5, b=5, l=5, r=5),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      uniformtext_minsize=12, uniformtext_mode='hide')

    return [
        html.Div(f"Sentiment: {sentiment_label}", style={'color': sentiment_color, 'margin-bottom': '10px'}),
        html.Div([html.Span(f"{word[0]}: {word[1]}", style={'margin-right': '10px'}) for word in insight_result], style={'margin-bottom': '10px'}),
        html.Div(f"Total Words: {len(filtered_tokens)}", style={'margin-bottom': '10px'})
    ], df






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

# Define the app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Update page content based on the URL
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)








def display_page(pathname):
    if pathname == '/tools':
        return master_layout()
    elif pathname == '/talk_to_data':
        return smartdata_layout
    else:
        return default_layout

# Define the main app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

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



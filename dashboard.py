from flask import Flask, jsonify, send_from_directory
import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px
import os

# Flask App
app = Flask(__name__)

data_file = "fraud_data.csv"

def load_data():
    return pd.read_csv(data_file)

@app.route("/api/summary")
def summary():
    df = load_data()
    total_transactions = len(df)
    fraud_cases = df[df['is_fraud'] == 1].shape[0]
    fraud_percentage = (fraud_cases / total_transactions) * 100
    return jsonify({
        "total_transactions": total_transactions,
        "fraud_cases": fraud_cases,
        "fraud_percentage": fraud_percentage
    })

@app.route("/api/fraud_trends")
def fraud_trends():
    df = load_data()
    df['date'] = pd.to_datetime(df['date'])
    trend_data = df[df['is_fraud'] == 1].groupby(df['date'].dt.date).size().reset_index(name='count')
    return jsonify(trend_data.to_dict(orient='records'))

@app.route("/api/fraud_location")
def fraud_location():
    df = load_data()
    fraud_data = df[df['is_fraud'] == 1][['latitude', 'longitude']]
    return jsonify(fraud_data.to_dict(orient='records'))

@app.route("/api/fraud_by_device")
def fraud_by_device():
    df = load_data()
    fraud_data = df[df['is_fraud'] == 1]['device'].value_counts().reset_index()
    fraud_data.columns = ['device', 'count']
    return jsonify(fraud_data.to_dict(orient='records'))

@app.route("/api/fraud_by_browser")
def fraud_by_browser():
    df = load_data()
    fraud_data = df[df['is_fraud'] == 1]['browser'].value_counts().reset_index()
    fraud_data.columns = ['browser', 'count']
    return jsonify(fraud_data.to_dict(orient='records'))

# Dash App
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/')

dash_app.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),
    
    # Summary
    html.Div(id='summary-stats'),
    
    # Fraud Trends
    dcc.Graph(id='fraud-trends'),
    
    # Fraud Locations
    dcc.Graph(id='fraud-map'),
    
    # Fraud by Device & Browser
    dcc.Graph(id='fraud-by-device'),
    dcc.Graph(id='fraud-by-browser')
])

@dash_app.callback(
    dash.dependencies.Output('summary-stats', 'children'),
    [dash.dependencies.Input('fraud-trends', 'id')]
)
def update_summary(_):
    response = pd.read_json("http://localhost:5000/api/summary")
    return html.Div([
        html.P(f"Total Transactions: {response['total_transactions'][0]}", className='summary-box'),
        html.P(f"Fraud Cases: {response['fraud_cases'][0]}", className='summary-box'),
        html.P(f"Fraud Percentage: {response['fraud_percentage'][0]:.2f}%", className='summary-box')
    ])

@dash_app.callback(
    dash.dependencies.Output('fraud-trends', 'figure'),
    [dash.dependencies.Input('fraud-trends', 'id')]
)
def update_trends(_):
    trend_data = pd.read_json("http://localhost:5000/api/fraud_trends")
    fig = px.line(trend_data, x='date', y='count', title='Fraud Cases Over Time')
    return fig

@dash_app.callback(
    dash.dependencies.Output('fraud-map', 'figure'),
    [dash.dependencies.Input('fraud-map', 'id')]
)
def update_map(_):
    location_data = pd.read_json("http://localhost:5000/api/fraud_location")
    fig = px.scatter_mapbox(location_data, lat='latitude', lon='longitude',
                             title='Fraud Cases by Location', zoom=2, mapbox_style="carto-positron")
    return fig

@dash_app.callback(
    dash.dependencies.Output('fraud-by-device', 'figure'),
    [dash.dependencies.Input('fraud-by-device', 'id')]
)
def update_device_chart(_):
    device_data = pd.read_json("http://localhost:5000/api/fraud_by_device")
    fig = px.bar(device_data, x='device', y='count', title='Fraud Cases by Device')
    return fig

@dash_app.callback(
    dash.dependencies.Output('fraud-by-browser', 'figure'),
    [dash.dependencies.Input('fraud-by-browser', 'id')]
)
def update_browser_chart(_):
    browser_data = pd.read_json("http://localhost:5000/api/fraud_by_browser")
    fig = px.bar(browser_data, x='browser', y='count', title='Fraud Cases by Browser')
    return fig

if __name__ == '__main__':
    app.run(debug=True)

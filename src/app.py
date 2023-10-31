import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
import pandas as pd
import plotly.express as px
from os.path import join
import re
from glob import glob

DATA_PATH = join('..', 'data')

pattern = re.compile(r'^av.*\.csv$', re.IGNORECASE)
files = [f for f in glob('*', root_dir=DATA_PATH) if pattern.match(f)]

example=join(DATA_PATH, files[0])

# Read the CSV file
df = pd.read_csv(join(DATA_PATH, files[0]), header=0)


# Initialize the app
app = dash.Dash(__name__)

# Create the layout
app.layout = html.Div([
    html.H1("Interactive Dashboard"),

    # Line plot with markers
    dcc.Graph(id='line-plot'),

    # Button to delete selected points
    html.Button('Delete Selected Points', id='delete-button'),

    # Hidden div to store selected points
    html.Div(id='selected-points', style={'display': 'none'}),
])

# Callbacks for line plot, selected points, and delete button
@app.callback(
    [Output('line-plot', 'figure'), Output('selected-points', 'children')],
    [Input('line-plot', 'selectedData'), Input('delete-button', 'n_clicks')],
    [State('selected-points', 'children')]
)
def update_line_plot(selected_data, n_clicks, stored_selected_points):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'line-plot' and selected_data:
        selected_indices = [point['pointIndex'] for point in selected_data['points']]
        selected_points = df.loc[selected_indices]
        if stored_selected_points:
            stored_selected_points = pd.read_json(stored_selected_points)
            selected_points = pd.concat([stored_selected_points, selected_points])
        return update_fig(selected_points), selected_points.to_json()

    if trigger_id == 'delete-button' and n_clicks and stored_selected_points:
        selected_points = pd.read_json(stored_selected_points)
        df_without_selected = df.drop(selected_points.index)
        df_without_selected.to_csv(f"{example}", index=False)
        return update_fig(df_without_selected), ''

    return update_fig(df), ''

# Helper function to update the figure
def update_fig(data):
    fig = px.line(data, x='Displacement', y='Axial force', title='Axial Force vs. Displacement',
                  render_mode='webgl', markers=True)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)






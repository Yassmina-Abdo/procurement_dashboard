import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import dash_table
from dash_table import DataTable, FormatTemplate
import pandas as pd
from dash.dependencies import Input, Output

ecom_sales = pd.read_csv('/usr/local/share/datasets/ecom_sales.csv')
logo_link = 'https://assets.datacamp.com/production/repositories/5893/datasets/fdbe0accd2581a0c505dab4b29ebb66cf72a1803/e-comlogo.png'



major_categories = list(ecom_sales['Major Category'].unique())
large_tb = ecom_sales.groupby(['Major Category', 'Minor Category'])['OrderValue'].agg(
    ['sum', 'count', 'mean']).reset_index().rename(
    columns={'count': 'Sales Volume', 'sum': 'Total Sales ($)', 'mean': 'Average Order Value ($)'})

ecom_country = ecom_sales.groupby('Country')['OrderValue'].agg('sum').reset_index(name='Total Sales ($)')
bar_fig_country = px.bar(ecom_country, x='Total Sales ($)', y='Country', width=500, height=450,
                         title='Total Sales by Country (Hover to filter the Minor Category bar chart!)',
                         custom_data=['Country'], color='Country',
                         color_discrete_map={'United Kingdom': 'lightblue', 'Germany': 'orange', 'France': 'darkblue',
                                             'Australia': 'green', 'Hong Kong': 'red'})

money_format = FormatTemplate.money(2)
money_cols = ['Total Sales ($)', 'Average Order Value ($)']


'''
d_columns = [{'name': x, 'id': x} for x in large_tb.columns if x not in money_cols]
d_columns += [
    {'name': 'Total Sales ($)', 'id': 'Total Sales ($)',
     'type': 'numeric',
     'format': money_format
     },
    {'name': 'Average Order Value ($)', 'id': 'Average Order Value ($)',
     'type': 'numeric',
     'format': money_format}]

d_table = DataTable(
    # Set up the columns and data
    columns=d_columns,
    data=large_tb.to_dict('records'),
    cell_selectable=False,
    # Set up sort, filter and pagination
    sort_action='native',
    filter_action='native',
    page_action='native',
    page_current=0,
    page_size=7,
) 
'''




app = dash.Dash(__name__)

app.layout = html.Div([
    html.Img(src=logo_link,
             style={'margin': '30px 0px 0px 0px'}),
    html.H1('Sales breakdowns'),
    html.Div(
        children=[
            html.Div(
                children=[
                    html.H2('Controls'),
                    html.Br(),
                    html.H3('Major Category Select'),
                    dcc.Dropdown(id='major_cat_dd',
                                 options=[{'label': category, 'value': category} for category in major_categories],
                                 style={'width': '200px', 'margin': '0 auto'}),
                    html.Br(),
                    html.H3('Minor Category Select'),
                    dcc.Dropdown(id='minor_cat_dd',
                                 style={'width': '200px', 'margin': '0 auto'})
                ],
                style={'width': '350px', 'height': '360px', 'display': 'inline-block', 'vertical-align': 'top',
                       'border': '1px solid black', 'padding': '20px'}),
            html.Div(children=[
                html.H3(id='chosen_major_cat_title'),
                dcc.Graph(id='sales_line')
            ],
                style={'width': '700px', 'height': '380px', 'display': 'inline-block', 'margin-bottom': '5px'}
            )
        ]),

    '''
    html.Div(
        # Insert the Data Table
        d_table,
        style={'width': '1000px', 'height': '350px', 'margin': '10px auto', 'padding-right': '30px'}),
    '''

    ,html.Div(children=[
        html.Div(dcc.Graph(id='major_cat', figure=bar_fig_country), style={'display': 'inline-block'}),
        html.Div(dcc.Graph(id='minor_cat'), style={'display': 'inline-block'})
    ],
        style={'width': '1000px', 'height': '650px', 'display': 'inline-block'}
    ),
]
    , style={'text-align': 'center', 'display': 'inline-block', 'width': '100%'}
)








# Set up a callback with multiple outputs
@app.callback(
    Output('minor_cat_dd', 'options'),
    Output('chosen_major_cat_title', 'children'),
    Input('major_cat_dd', 'value'))
def update_dd(major_cat_dd):
    major_minor = ecom_sales[['Major Category', 'Minor Category']].drop_duplicates()
    relevant_minor = major_minor[major_minor['Major Category'] == major_cat_dd]['Minor Category'].values.tolist()
    minor_options = [dict(label=x, value=x) for x in relevant_minor]

    if not major_cat_dd:
        major_cat_dd = 'ALL'

    major_cat_title = f'This is in the Major Category of : {major_cat_dd}'

    return minor_options, major_cat_title


@app.callback(
    Output('sales_line', 'figure'),
    Input('minor_cat_dd', 'value'))
def update_line(minor_cat):
    minor_cat_title = 'All'
    ecom_line = ecom_sales.copy()
    if minor_cat:
        minor_cat_title = minor_cat
        ecom_line = ecom_line[ecom_line['Minor Category'] == minor_cat]
    ecom_line = ecom_line.groupby('Year-Month')['OrderValue'].agg('sum').reset_index(name='Total Sales ($)')
    line_graph = px.line(ecom_line, x='Year-Month', y='Total Sales ($)',
                         title=f'Total Sales by Month for Minor Category: {minor_cat_title}', height=350)

    return line_graph


@app.callback(
    Output('minor_cat', 'figure'),
    Input('major_cat', 'hoverData'))
def update_min_cat_hover(hoverData):
    hover_country = 'Australia'

    if hoverData:
        hover_country = hoverData['points'][0]['customdata'][0]

    minor_cat_df = ecom_sales[ecom_sales['Country'] == hover_country]
    minor_cat_agg = minor_cat_df.groupby('Minor Category')['OrderValue'].agg('sum').reset_index(name='Total Sales ($)')
    ecom_bar_minor_cat = px.bar(minor_cat_agg, x='Total Sales ($)', y='Minor Category', orientation='h', height=450,
                                width=480, title=f'Sales by Minor Category for: {hover_country}')
    ecom_bar_minor_cat.update_layout({'yaxis': {'dtick': 1, 'categoryorder': 'total ascending'}, 'title': {'x': 0.5}})

    return ecom_bar_minor_cat




if __name__ == '__main__':
    app.run_server(debug=True)
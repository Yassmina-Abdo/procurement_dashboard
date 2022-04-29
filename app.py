# %% md
# Imports
# %%
import jupyter_dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input, Output
from dash_table import DataTable, FormatTemplate
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd

# %% md
# Data Exploration
# %%
df = pd.read_csv('contracts_data.csv')
df.head()
# %%
df.info()
# %%
# precentage of nulls in each col
(df.isnull().sum() / df['Award Date'].count()) * 100
# %%
# contract consulting in diffrent names
list(df['Commodity Category'].unique())
# %% md
# Data Preprocessing
# %%
# Replace Consult Col
df['Commodity Category'].replace('CONTRACT CONSULTANTS', 'CONSULTING', inplace=True)
df['Commodity Category'].replace('CONSULTING CONTRACT', 'CONSULTING', inplace=True)
df['Commodity Category'].replace('CONTRACT CONSULTING', 'CONSULTING', inplace=True)
# %%
# give Supplier Country Code its value
df.loc[16, 'Supplier Country Code'] = 'NAM'

# drop Selection Number whih i won't need
df = df.drop('Selection Number', axis=1)

# remove outliars
q_low = df['Contract Award Amount'].quantile(0.01)
q_hi = df['Contract Award Amount'].quantile(0.99)

df = df[(df['Contract Award Amount'] < q_hi) & (df['Contract Award Amount'] > q_low)]

# impute depend on (country and position)
rr = df.groupby(['Commodity Category']).mean()['Contract Award Amount']
for i in rr.index:
    if (df[df['Commodity Category'] == i]['Contract Award Amount'].isna().sum()) != 0:
        r = df[df['Commodity Category'] == i]['Contract Award Amount'].fillna(rr[i])
        df.loc[r.index, 'Contract Award Amount'] = r

# %%
# date conversion
import datetime


def convertDate(d):
    new_date = datetime.datetime.strptime(d, "%Y-%m-%dT%H:%M:%S.%f")
    return new_date.date()


df['Dates'] = df['Award Date'].apply(convertDate)
df['Dates'] = pd.to_datetime(df['Dates'])


# %%
def process_date(df):
    date_parts = ['year', 'week', 'month', 'weekofyear', 'day', 'quarter']
    for part in date_parts:
        part_col = 'award_date' + '_' + part
        df[part_col] = getattr(df['Dates'].dt, part).astype(int)
    return df


process_date(df)
# %%

# get the most frequent countries
countries = {}
for i in (df['Supplier Country'].unique()):
    countries[i] = (df['Supplier Country'] == i).sum()

countries = dict(sorted(countries.items(), key=lambda x: x[1], reverse=True))
countries = pd.Series(countries)
countries = countries[:5]

# get the most frequent category
categories = {}
for i in (df['Commodity Category'].unique()):
    categories[i] = (df['Commodity Category'] == i).sum()

categories = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))
categories = pd.Series(categories)
categories = categories[:5]

# Data Frame have the most Freq Countries and Categories
most_freqcountr_df = pd.DataFrame(columns=df.columns)
for i in countries.index:
    df_temp = df[df['Supplier Country'] == i]
    most_freqcountr_df = most_freqcountr_df.append(df_temp)

most_freqcat_df = pd.DataFrame(columns=df.columns)
for i in categories.index:
    df_temp = most_freqcountr_df[most_freqcountr_df['Commodity Category'] == i]
    most_freqcat_df = most_freqcat_df.append(df_temp)

# Convert col To Float
most_freqcat_df['Contract Award Amount'] = most_freqcat_df['Contract Award Amount'].astype(float)
most_freq_df = most_freqcat_df.copy()

##################
# sort by year
most_freq_df = most_freq_df.sort_values(by=['award_date_year'], ascending=True)
df = df.sort_values(by=['award_date_year'], ascending=True)
###################

# Get less Description
for i in range(len(most_freq_df)):
    splitted = most_freq_df['Contract Description'].iloc[i].split()[0]
    most_freq_df['Contract Description'].iloc[i] = splitted
# Handle some issues in col Contract Description
most_freq_df = most_freq_df.reset_index().drop('index', axis=1)
most_freq_df['Contract Description'].iloc[198] = 'Launch Dev'
most_freq_df['Contract Description'].iloc[199] = 'Launch Dev'
most_freq_df['Contract Description'].iloc[226] = 'Pakistan'
# most_freq_df['Contract Description'].iloc[1061] = 'Somalia'
# %%
most_freq_df.head()
# %% md
# Insights
# %%
# All Unique Categories
major_categories = list(most_freq_df['Commodity Category'].unique())

# Group Category and its minor then do some calculations
large_tb = most_freq_df.groupby(['Commodity Category', 'Contract Description'])['Contract Award Amount'].agg(
    ['sum', 'count', 'mean']).reset_index().rename(
    columns={'count': 'Contract Volume', 'sum': 'Total Contracts ($)', 'mean': 'Average Contract Value ($)'})

money_format = FormatTemplate.money(2)
money_cols = ['Total Contracts ($)', 'Average Contract Value ($)']

# Contracts per Country
contracts_country = most_freq_df.groupby('Supplier Country')['Contract Award Amount'].agg('sum').reset_index(
    name='Total Contracts ($)')

# %% md
# Dashboard
# %% md
# Graphs
# %%
# Bar Graph For Total contracts per Country
bar_fig_country = px.bar(contracts_country, x='Total Contracts ($)', y='Supplier Country',
                         title='Total Contracts by Country (Hover to filter)',
                         custom_data=['Supplier Country'], color='Supplier Country',
                         color_discrete_map={'United Kingdom': '#FFA630', 'India': '#D7E8BA', 'France': '#4DA1A9',
                                             'USA': '#611C35', 'Netherlands': '#CE7DA5', 'Germany': '#E05263',
                                             'Spain': '#FDF6E3', 'Kenya': 'magenta', 'Belgium': 'royalblue',
                                             'Canada': 'lightcyan'})
bar_fig_country.update_layout(
    {'font': {'color': '#FFFFFF'}, 'plot_bgcolor': 'rgb(0, 43, 54)', 'paper_bgcolor': 'rgb(0, 43, 54)'})


# %%
def create_donut(title, colname):
    fig = go.Figure(
        data=[go.Pie(labels=[title], hole=0.8, textinfo="none", hoverinfo='none', marker={'colors': ['#B58900']})])

    fig.add_annotation(text=str(len(df[colname].unique())) + "<br>" + "<br>" + title, x=0.5, y=0.5, showarrow=False,
                       font={'family': "Times", 'size': 31, 'color': '#FFFFFF'})

    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', "showlegend": False,
                       'margin': dict(r=0, l=0, t=0, b=0), 'width': 200, 'height': 300})

    return (dcc.Graph(figure=fig))


# %% md
# Components
# %%
def create_supplier_dopdown():
    return (dcc.Dropdown(id='user_choice', options=[{'label': sup, "value": sup} for sup in df.Supplier.unique()],
                         value=2007, clearable=False)
            )


# %%
def create_card(graph):
    card = dbc.Card(
        dbc.CardBody(
            [
                html.Div(graph),
            ]
        ),
        color="dark",
        inverse=True,
        outline=False,
    )
    return (card)


# %% md
# Cards Rows
# %%

card1 = create_card(create_donut('Categories', 'Commodity Category'))
card2 = create_card(create_donut('Suppliers', 'Supplier'))
card3 = create_card(create_donut('Countries', 'Supplier Country'))
card4 = create_card((dcc.Graph(id='major_cat', figure=bar_fig_country)))
card5 = create_card((dcc.Graph(id='minor_cat')))
card6 = create_card((dcc.Graph(id='histogram_graph')))
card7 = create_card((dcc.Graph(id='line_graph')))
card8 = create_card((dcc.Graph(id='count_graph')))

Row0 = dbc.Row(
    [dbc.Col([html.H4()]),
     html.H1("\t" + 'Procurement Department', className="title",
             style={'textAlign': 'center', 'color': '#B58900', 'size': 24, 'family': "Times"}),
     html.Br(),
     html.Br(),
     html.Br()
     ])
Row1 = dbc.Row(
    [dbc.Col([html.H6()]),
     dbc.Col(id='card1', children=[card1]),
     dbc.Col(id='card2', children=[card2]),
     dbc.Col(id='card3', children=[card3]),
     dbc.Col([html.H6()])
     ])
Row2 = dbc.Row(
    [
        html.Br()
    ])
Row3 = dbc.Row(
    [
        dbc.Col(id='card4', children=[card4]),
        dbc.Col(id='card5', children=[card5]),
    ], justify='around'
)

Row4 = dbc.Row([
    html.Br(),
    html.Br(),
    html.H4('Contract Amount Per Year', style={'textAlign': 'left', 'color': '#B58900', 'size': 50}),
    dcc.Dropdown(
        id="histogram_dropdown",
        options=most_freq_df.award_date_year.unique(),
        value=2020,
        clearable=False, style={'width': 600}
    ),
    html.Br(),
    html.Br(),
    Row2,
    dbc.Col(id='card6', children=[card6])
], justify='around')

Row5 = dbc.Row([
    html.Br(),
    html.Br(),
    html.H4('Contract Amount For Each Country', style={'textAlign': 'left', 'color': '#B58900', 'size': 50}),
    dcc.Dropdown(
        id="country",
        options=df['Supplier Country'].unique(),
        value="USA",
        clearable=False, style={'width': 600}
    ),
    html.Br(),
    html.Br(),
    Row2,
    dbc.Col(id='card7', children=[card7]),
    dbc.Col(id='card8', children=[card8]),

], justify='around')

# %%
app = jupyter_dash.JupyterDash(__name__, external_stylesheets=[dbc.themes.SOLAR])
app.title = 'Procurement Dashboard'
# %% md
# App Layout
# %%
app.layout = html.Div([
    Row0,
    Row1,
    Row2,
    Row3,
    Row5,
    Row2,
    Row4

])


# %% md
# Callbacks Functions
# %%
# bar graph update for major and minor Category
@app.callback(
    Output('minor_cat', 'figure'),
    Input('major_cat', 'hoverData'))
def update_min_cat_hover(hoverData):
    hover_country = 'USA'

    if hoverData:
        hover_country = hoverData['points'][0]['customdata'][0]

    minor_cat_df = most_freq_df[most_freq_df['Supplier Country'] == hover_country]
    minor_cat_agg = minor_cat_df.groupby('Contract Description')['Contract Award Amount'].agg('sum').reset_index(
        name='Total Contracts ($)')
    contr_bar_minor_cat = px.bar(minor_cat_agg, x=minor_cat_agg['Total Contracts ($)'].iloc[:20],
                                 y=minor_cat_agg['Contract Description'].iloc[:20], orientation='h',
                                 title=f'Contracts by Minor Category for: {hover_country}')
    contr_bar_minor_cat.update_layout(
        {'font': {'color': '#FFFFFF'}, 'plot_bgcolor': 'rgb(0, 43, 54)', 'paper_bgcolor': 'rgb(0, 43, 54)',
         'yaxis': {'dtick': 1, 'categoryorder': 'total ascending'}, 'title': {'x': 0.5},
         'xaxis_title': 'Total Contracts ($)', 'yaxis_title': 'Minor Category'})
    contr_bar_minor_cat.update_traces({'marker_color': 'rgb(145, 47, 86)'})

    return contr_bar_minor_cat


# %%
# Year as an input then return sum amounts for each category
@app.callback(
    Output("histogram_graph", 'figure'),
    Input("histogram_dropdown", "value"))
def update_bar_chart(year):
    mask = most_freq_df["award_date_year"] == year
    fig = px.histogram(most_freq_df[mask], x="Commodity Category", y="Contract Award Amount",
                       color="Supplier Country",
                       barmode="group",
                       color_discrete_map={'United Kingdom': '#FFA630', 'India': '#D7E8BA', 'France': '#4DA1A9',
                                           'USA': '#611C35', 'Netherlands': '#CE7DA5', 'Germany': '#E05263'})

    fig.update_layout(
        {'font': {'color': '#FFFFFF'}, 'plot_bgcolor': 'rgb(0, 43, 54)', 'paper_bgcolor': 'rgb(0, 43, 54)'})
    return fig


# %%
##Country as an input then return sum amounts for each year
@app.callback(
    Output("line_graph", "figure"),
    Input("country", "value"))
def update_bar_chart(country):
    mask = 'USA'
    if country:
        mask = df["Supplier Country"] == country

    dff = df[mask]
    fig = px.line(dff, x=dff['award_date_year'].unique(),
                  y=dff.groupby(['award_date_year']).sum()['Contract Award Amount'],
                  text=((dff.groupby(['award_date_year']).sum()['Contract Award Amount']).astype(
                      float) / 1000000).round(3), title='Total Amount Per Year')

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(
        {'font': {'color': '#FFFFFF'}, 'plot_bgcolor': 'rgb(0, 43, 54)', 'paper_bgcolor': 'rgb(0, 43, 54)',
         'xaxis_title': 'Contract Year', 'yaxis_title': 'Total Contracts ($)'})
    return fig


# %%
# take country as input --> return percentage of each category
@app.callback(
    Output("count_graph", "figure"),
    Input("country", "value"))
def update_bar_chart(country):
    mask = df["Supplier Country"] == country
    dff = df[mask]
    fig = go.Figure(go.Bar(
        y=(dff.groupby('Commodity Category').nunique()['Supplier']).sort_values(ascending=False)[0:6].index,
        x=(dff.groupby('Commodity Category').nunique()['Supplier']).sort_values(ascending=False)[0:6],
        text=((((dff.groupby('Commodity Category').nunique()['Supplier']).sort_values(ascending=False)[0:6]) / len(
            dff['Supplier'].unique())) * 100).apply(lambda x: '{0:1.2f}%'.format(x)),
        orientation='h',
        marker=dict(color='rgba(246, 78, 139, 0.6)',
                    line=dict(color='rgba(246, 78, 139, 1.0)', width=3))
    ))

    fig.update_xaxes(showgrid=False)
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(
        {'font': {'color': '#FFFFFF'}, 'plot_bgcolor': 'rgb(0, 43, 54)', 'paper_bgcolor': 'rgb(0, 43, 54)'},
        barmode='stack')
    fig.update_layout(title='categories percentage', xaxis_tickfont_size=14,
                      yaxis=dict(title='Categories', titlefont_size=16, tickfont_size=14, ),
                      xaxis=dict(
                          title='count',
                          titlefont_size=16,
                          tickfont_size=14,
                      )
                      )
    return fig


# %% md
# Run Server
# %%
if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
from dash_table.Format import Format, Group, Scheme
import dash_table.FormatTemplate as FormatTemplate
import plotly.express as px

import pandas as pd
import numpy as np
from datetime import date


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server


#####################
# Empty row

def get_emptyrow(h='45px'):
    """This returns an empty row of a defined height"""

    emptyrow = html.Div([
        html.Div([
            html.Br()
        ], className = 'col-12')
    ],
    className = 'row',
    style = {'height' : h})

    return emptyrow

####################### Corporate css formatting
corporate_colors = {
    'dark-blue-grey' : 'rgb(62, 64, 76)',
    'medium-blue-grey' : 'rgb(77, 79, 91)',
    'superdark-green' : 'rgb(41, 56, 55)',
    'dark-green' : 'rgb(57, 81, 85)',
    'medium-green' : 'rgb(93, 113, 120)',
    'light-green' : 'rgb(186, 218, 212)',
    'pink-red' : 'rgb(255, 101, 131)',
    'dark-pink-red' : 'rgb(247, 80, 99)',
    'white' : 'rgb(251, 251, 252)',
    'light-grey' : 'rgb(208, 206, 206)',
	"green-dark" : 'rgb(68,101, 92)'
}

externalgraph_rowstyling = {
    'margin-left' : '15px',
    'margin-right' : '15px'
}

externalgraph_colstyling = {
    'border-radius' : '10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : corporate_colors['superdark-green'],
    'background-color' : corporate_colors['superdark-green'],
    'box-shadow' : '0px 0px 17px 0px rgba(186, 218, 212, .5)',
    'padding-top' : '10px'
}

filterdiv_borderstyling = {
    'border-radius' : '0px 0px 10px 10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : corporate_colors['light-green'],
    'background-color' : corporate_colors['light-green'],
    'box-shadow' : '2px 5px 5px 1px rgba(255, 101, 131, .5)'
    }

recapdiv = {
    'border-radius' : '10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : 'rgb(251, 251, 252, 0.1)',
    'margin-left' : '15px',
    'margin-right' : '15px',
    'margin-top' : '15px',
    'margin-bottom' : '15px',
    'padding-top' : '5px',
    'padding-bottom' : '5px',
    'background-color' : 'rgb(251, 251, 252, 0.1)'
    }

recapdiv_text = {
    'text-align' : 'left',
    'font-weight' : '350',
    'color' : corporate_colors['white'],
    'font-size' : '1.5rem',
    'letter-spacing' : '0.04em'
    }

####################### Corporate chart formatting

corporate_title = {
    'font' : {
        'size' : 16,
        'color' : corporate_colors['white']}
}

corporate_xaxis = {
    'showgrid' : False,
    'linecolor' : corporate_colors['light-grey'],
    'color' : corporate_colors['light-grey'],
    'tickangle' : 315,
    'titlefont' : {
        'size' : 12,
        'color' : corporate_colors['light-grey']},
    'tickfont' : {
        'size' : 11,
        'color' : corporate_colors['light-grey']},
    'zeroline': False
}

corporate_yaxis = {
    'showgrid' : True,
    'color' : corporate_colors['light-grey'],
    'gridwidth' : 0.5,
    'gridcolor' : corporate_colors['dark-green'],
    'linecolor' : corporate_colors['light-grey'],
    'titlefont' : {
        'size' : 12,
        'color' : corporate_colors['light-grey']},
    'tickfont' : {
        'size' : 11,
        'color' : corporate_colors['light-grey']},
    'zeroline': False
}

corporate_font_family = 'Dosis'

corporate_legend = {
    'orientation' : 'h',
    'yanchor' : 'bottom',
    'y' : 1.01,
    'xanchor' : 'right',
    'x' : 1.05,
	'font' : {'size' : 9, 'color' : corporate_colors['light-grey']}
} # Legend will be on the top right, above the graph, horizontally

corporate_margins = {'l' : 5, 'r' : 5, 't' : 45, 'b' : 15}  # Set top margin to in case there is a legend


# DATA
df = pd.read_csv('./data/data_tfm.csv')
#data_df=df.to_dict('records')
#columns=[{"name": i, "id": i} for i in df.columns]
PAGE_SIZE = 10
# LAYOUT
app.layout = html.Div([

    #####################
    #Row 1 : Header
    html.Div([

        html.Div([], className = 'col-2'), #Same as img width, allowing to have the title centrally aligned

        html.Div([
            html.H1(children='Performance Dashboard DBB -- TFM',
                    style = {'textAlign' : 'center'}
            )],
            className='col-8',
            style = {'padding-top' : '1%'}
        ),

        html.Div([
            html.Img(
                    src = 'https://www.oqotech.com/wp-content/uploads/2018/03/logo-CHS.jpg',
                    height = '60 px',
                    width = 'auto')
            ],
            className = 'col-2',
            style = {
                    'align-items': 'center',
                    'padding-top' : '1%',
                    'height' : 'auto'})

        ],
        className = 'row',
        style = {'height' : '4%',
                'background-color' : corporate_colors['green-dark']}
        ),
	######################
	#Row2 : KPI
	
	html.Div([
        html.Div([
			html.H5(
                children='Error Rate % :',
                style = {'font-size': '20px','text-align' : 'center', 'color' : corporate_colors['light-green']}
            ),
            html.Img(src="https://cdn.pixabay.com/photo/2013/07/12/12/40/abort-146096_1280.png",
                     style={"width": "50px"}),
            html.H2(
                id='error_rate',
            )
        ]),
        html.Div([
			html.H5(
                children='Nº Samples :',
                style = {'font-size': '20px','text-align' : 'center', 'color' : corporate_colors['light-green']}
            ),
            html.Img(src="https://assets.wprock.fr/emoji/joypixels/512/1f522.png",
                     style={"width": "50px"}),
            html.H2(
                id='num_samples',
            )
        ]),
        html.Div([
			html.H5(
                children='Throughput :',
                style = {'font-size': '20px','text-align' : 'center', 'color' : corporate_colors['light-green']}
            ),
            html.Img(src="https://assets.wprock.fr/emoji/joypixels/512/1f4c8.png",
                     style={"width": "50px"}),
            html.H2(
                id='throughput',
            )
        ]),
    ], style={"columnCount": 3, 'textAlign': "center"}),
	
	html.Div([
        html.Div([
			html.H5(
                children='AWS Task :',
                style = {'font-size': '20px','text-align' : 'center', 'color' : corporate_colors['light-green']}
            ),
            html.H2(
                id='aws_task',
            )
        ]),
        html.Div([
			html.H5(
                children='Nº Threads :',
                style = {'font-size': '20px','text-align' : 'center', 'color' : corporate_colors['light-green']}
            ),
            html.H2(
                id='num_threads',
            )
        ]),
        html.Div([
			html.H5(
                children='Task CPU :',
                style = {'font-size': '20px','text-align' : 'center', 'color' : corporate_colors['light-green']}
            ),
            html.H2(
                id='task_CPU',
            )
        ]),
		html.Div([
			html.H5(
                children='Task Memory :',
                style = {'font-size': '20px','text-align' : 'center', 'color' : corporate_colors['light-green']}
            ),
            html.H2(
                id='task_memory',
            )
        ])
    ], style={"columnCount": 4, 'textAlign': "center"}),
	#####################
    #Row 3 : Filters


    html.Div([ # External row

        html.Div([ # External 12-column

            html.Div([ # Internal row
			
                #Internal columns
                html.Div([
                ],
                className = 'col-2'), # Blank 2 columns

                #Filter pt 1
                html.Div([

                    html.Div([
                        html.H5(
                            children='Filters by Endpoint:',
                            style = {'font-size': '20px','text-align' : 'left', 'color' : corporate_colors['medium-blue-grey']}
                        ),
						
                        #Date range picker
                        html.Div(['Select desired endpoint: ',
								dcc.RadioItems(
									id='endpoint_radio',
									options=[{"label": endpoint, "value": endpoint} for endpoint in df.endpoint.unique()],
									#value=[endpoint for endpoint in df.endpoint.unique()],
									value= 'device',
									labelStyle={'display': 'inline-block'},
									style = {'font-size': '20px', 'color' : corporate_colors['medium-blue-grey'], 'white-space': 'nowrap', 'text-overflow': 'ellipsis'})
									], style = {'margin-top' : '5px'}
                         )

                    ],
                    style = {'margin-top' : '10px',
                            'margin-bottom' : '5px',
                            'text-align' : 'left',
                            'paddingLeft': 5})

                ],
                className = 'col-4'), # Filter part 1

                #Filter pt 2
                html.Div([

                    html.Div([
                        html.H5(
                            children='Filters by Graph:',
                            style = {'font-size': '20px','text-align' : 'left', 'color' : corporate_colors['medium-blue-grey']}
                        ),
                        #Reporting group selection l1
                        html.Div([
                            dcc.Dropdown(id = 'graph_dropdown',
                                options=[{"label": i, "value": i} for i in ['ScatterPlot', 'BoxPlot', 'BarPlot']],
								placeholder = "Select prefered graph",
								multi=False,
                                value = 'BoxPlot',
                                style = {'font-size': '20px', 'color' : corporate_colors['medium-blue-grey'], 'white-space': 'nowrap', 'text-overflow': 'ellipsis'}
                                )
                            ],
                            style = {'width' : '70%', 'margin-top' : '5px'})
                    ],
                    style = {'margin-top' : '10px',
                            'margin-bottom' : '5px',
                            'text-align' : 'left',
                            'paddingLeft': 5})

                ],
                className = 'col-4'), # Filter part 2
				html.Div([
                ],
                className = 'col-2') # Blank 2 columns

            ],
            className = 'row') # Internal row
			

        ],
        className = 'col-12',
        style = filterdiv_borderstyling) # External 12-column

    ],
    className = 'row sticky-top'), # External row
    #####################
    #Row 4
	get_emptyrow(),
	#Datatable
	html.Div([
		html.Label('DataTable Values',
					style = {'font-size': '20px','text-align' : 'left', 'color' : corporate_colors['light-green']}
		),
		dash_table.DataTable(
					id='table-values',
					style_header = {
                            'backgroundColor': 'transparent',
                            'fontFamily' : corporate_font_family,
                            'font-size' : '1rem',
                            'color' : corporate_colors['light-green'],
                            'border': '0px transparent',
                            'textAlign' : 'center'},
					page_current= 0,
					page_size= PAGE_SIZE,
					page_action='custom',
					
					filter_action="custom", 
					filter_query = ''

		),
			html.Div(id='datatable-row-ids-container')

    ],style={"columnCount": 1, 'textAlign': "center", "margin-top": "24px", "margin-bottom": "48px"}),

    #####################
    #Row 5 : Charts
    html.Div([ # External row

        html.Div([
        ],
        className = 'col-1'), # Blank 1 column

        html.Div([ # External 10-column

            html.H2(children = "Testing Performances",
                    style = {'color' : corporate_colors['white']}),

            html.Div([ # Internal row

                #Chart Column
                html.Div([
                    dcc.Graph(
                        id='graph1')
                ],
                className = 'col-12')

            ],
            className = 'row'), # Internal row
			
			html.Div([ # Internal row

                #Chart Column
                html.Div([
                    dcc.Graph(
                        id='graph2')
                ],
                className = 'col-12')

            ],
            className = 'row'), # Internal row

            html.Div([ # Internal row

                #Chart Column
                html.Div([
                    dcc.Graph(
                        id='graph3')
                ],
                className = 'col-12')

            ],
            className = 'row'), # Internal row
			
			html.Div([ # Internal row

                #Chart Column
                html.Div([
                    dcc.Graph(
                        id='graph4')
                ],
                className = 'col-12')

            ],
            className = 'row'), # Internal row


        ],
        className = 'col-12',
        style = externalgraph_colstyling), # External 10-column


    ],
    className = 'row',
    style = externalgraph_rowstyling
    ), # External row

])



operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]

def split_filter_part(filter_part, df):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]
                value_part = value_part.strip()
                value = value_part

                col_type = df.dtypes[name]
                if col_type == np.int64:
                    value = int(value) 
                if col_type == np.float:
                    value = float(value)

                # word operators need spaces after them in the filter string, but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3


  
@app.callback(
	[dash.dependencies.Output('table-values', 'data'), 
	dash.dependencies.Output('table-values', 'columns'),
	

	Output('error_rate', 'children'),
    Output('num_samples', 'children'),
    Output('throughput', 'children'),
	Output('aws_task', 'children'),
	Output('task_CPU', 'children'),
	Output('task_memory','children'),
	Output('num_threads', 'children'),
	Output('graph1', 'figure'),
	Output('graph2', 'figure'),
	Output('graph3','figure'),
	Output('graph4', 'figure')
	],
	dash.dependencies.Input('endpoint_radio', 'value'),
	dash.dependencies.Input('graph_dropdown', 'value'),	
	dash.dependencies.Input('table-values', 'filter_query'),
	dash.dependencies.Input('table-values', "page_current"),
    dash.dependencies.Input('table-values', "page_size"))
	
	
def update_chart(endpoint_radio, graph_dropdown, filter, page_current, page_size):
	filtering_expressions = filter.split(' && ')
	df['config'] = df.apply(lambda x: 'Num_threads: ' + str(x['num_threads']) + ' ' + 'Task_CPU: ' + str(x['task_CPU']) + ' ' + 'Task_memory: ' + str(x['task_memory']), axis=1)
	columns=[{"name": i, "id": i} for i in df.columns if i != 'config']
	dff = df
	dff = dff[dff['endpoint'] == endpoint_radio]
	for filter_part in filtering_expressions:
		col_name, operator, filter_value = split_filter_part(filter_part, dff)
		if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
			dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
		elif operator == 'contains':
			dff = dff.loc[dff[col_name] == filter_value]
	page = page_current
	size = page_size
	data = dff.iloc[page * size: (page + 1) * size].to_dict('records')
	legend_conf = dict(
				orientation="h",
				yanchor="bottom",
				y=0,
				xanchor="left",
				x=1
	)
	error_rate = '0'
	num_samples = '0'
	throughput = '0'
	
	if endpoint_radio == 'device':
		best_config_device = df[(df['endpoint'] == 'device') & (df['Error %'] == '0.000%')].sort_values(['aws_task'], ascending=[True]).sort_values(['Throughput'], ascending= False).iloc[3]
		
		num_samples = best_config_device['Samples']
		error_rate = best_config_device['Error %']
		throughput = best_config_device['Throughput']
		aws_task = best_config_device['aws_task']
		task_CPU = best_config_device['task_CPU']
		task_memory = best_config_device['task_memory']
		num_threads = best_config_device['num_threads']
		if graph_dropdown == 'BoxPlot' : 
			graph1 = px.box(data, x= "Error %",  title="Distribution of error")
			graph2 = px.box(data, x= "Throughput", title = "Distribution of throughput")
			graph3 = px.box(data, x= "Average", title = "Distribution of average time")
			graph4 = px.box(data, x= "Samples", title = "Distribution of number of samples")
		elif graph_dropdown == 'ScatterPlot':
			graph1 = px.scatter(data, x="Average", y="Throughput", color="config" , symbol = 'aws_task', title = "Average time depending on throughput")
			graph1.update_layout(legend=legend_conf, legend_title_text='Configuration')
			graph2 = px.scatter(data, x="task_CPU", y="Throughput", color="config", symbol = 'aws_task', title = "Number of task CPU depending on throughput")
			graph2.update_layout(legend=legend_conf, legend_title_text='Configuration')
			graph3 = px.scatter(data, x="task_memory", y="Throughput", color="config", symbol = 'aws_task', title = "Available memory depending on throughput")
			graph3.update_layout(legend=legend_conf, legend_title_text='Configuration')
			graph4 = px.scatter(data, x="num_threads", y="Throughput", color="config", symbol = 'aws_task', title = "Number of threads depending on throughput")
			graph4.update_layout(legend=legend_conf, legend_title_text='Configuration')
		elif graph_dropdown == 'BarPlot':
			graph1 = px.bar(data, x='aws_task', y='Throughput', color='config',pattern_shape = 'aws_task' )
			graph1.update_layout(legend=legend_conf, legend_title_text='Configuration')
			graph2 = px.bar(data, x='Error %', y='Throughput', color='config',pattern_shape = 'aws_task')
			graph2.update_layout(legend=legend_conf, legend_title_text='Configuration')
			graph3 = px.bar(data, x='task_memory', y='Throughput', color='config',pattern_shape = 'aws_task')
			graph3.update_layout(legend=legend_conf, legend_title_text='Configuration')
			graph4 = px.bar(data, x='task_CPU', y='Throughput', color='config',pattern_shape = 'aws_task')
			graph4.update_layout(legend=legend_conf, legend_title_text='Configuration')



	elif endpoint_radio == 'patient':
		best_config_patient = df[(df['endpoint'] == 'patient') & (df['Error %'] == '0.000%')].sort_values(['aws_task'], ascending=[True]).sort_values(['Throughput'], ascending= False).iloc[0]

		num_samples = best_config_patient['Samples']
		error_rate = best_config_patient['Error %']
		throughput = best_config_patient['Throughput']		
		aws_task = best_config_patient['aws_task']
		task_CPU = best_config_patient['task_CPU']
		task_memory = best_config_patient['task_memory']
		num_threads = best_config_patient['num_threads']
		if graph_dropdown == 'BoxPlot' : 
			graph1 = px.box(data, x= "Error %",  title="Distribution of error")
			graph2 = px.box(data, x= "Throughput", title = "Distribution of throughput")
			graph3 = px.box(data, x= "Average", title = "Distribution of average time")
			graph4 = px.box(data, x= "Samples", title = "Distribution of number of samples")
		elif graph_dropdown == 'ScatterPlot':
			graph1 = px.scatter(data, x="Average", y="Throughput", color="config", symbol = 'aws_task', title = "Average time depending on throughput")
			graph1.update_layout(legend=legend_conf,legend_title_text='Configuration')
			graph2 = px.scatter(data, x="task_CPU", y="Throughput", color="config", symbol = 'aws_task', title = "Number of task CPU depending on throughput")
			graph2.update_layout(legend=legend_conf, legend_title_text='Configuration')
			graph3 = px.scatter(data, x="task_memory", y="Throughput", color="config", symbol = 'aws_task', title = "Available memory depending on throughput")
			graph3.update_layout(legend=legend_conf, legend_title_text='Configuration')
			graph4 = px.scatter(data, x="num_threads", y="Throughput", color="config", symbol = 'aws_task', title = "Number of threads depending on throughput")
			graph4.update_layout(legend=legend_conf, legend_title_text='Configuration')
		elif graph_dropdown == 'BarPlot':
			graph1 = px.bar(data, x='aws_task', y='Throughput', color='config',pattern_shape = 'aws_task')
			graph1.update_layout(legend=legend_conf, legend_title_text='Configuration')
			graph2 = px.bar(data, x='Error %', y='Throughput', color='config',pattern_shape = 'aws_task')
			graph2.update_layout(legend=legend_conf, legend_title_text='Configuration')
			graph3 = px.bar(data, x='task_memory', y='Throughput', color='config',pattern_shape = 'aws_task')
			graph3.update_layout(legend=legend_conf, legend_title_text='Configuration')
			graph4 = px.bar(data, x='task_CPU', y='Throughput', color='config',pattern_shape = 'aws_task')
			graph4.update_layout(legend=legend_conf, legend_title_text='Configuration')
	else: data, columns, error_rate, num_samples, throughput, aws_task, task_CPU, task_memory, num_threads, graph1, graph2, graph3, graph4
		
	return data, columns, error_rate, num_samples, throughput, aws_task, task_CPU, task_memory, num_threads, graph1, graph2, graph3, graph4

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port="80")

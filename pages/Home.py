import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from badges import badges

dash.register_page(__name__,
                   path='/',
                   title='LLM Playground 🎮',
                   name='LLM Playground 🎮')

layout = html.Div(
    [
        html.Div([
            html.Img(src=dash.get_asset_url('logo.svg'), height="200px"),
        ], style={'display': 'flex',
                  'justify-content': 'center',
                  'align-items': 'center',
                  'text-align': 'center',
                  }),
        html.Div([
            html.Div([
                # Add your introduction here
                dcc.Markdown('''
                    
                    ''', style={'font-size': '1.2rem', 'font-weight': 'bold'})

            ], style={'display': 'flex',
                      'justify-content': 'center',
                      'align-items': 'center',
                      'text-align': 'center',
                      'margin-top': '2rem',
                      'width': '800px'
                      }),
        ], className='page-intro-outer-div'),
        html.Div([
            html.Div([badges], style={'text-align': 'center'}),
        ], style={'width': '100%', 'position': 'fixed', 'bottom': '20px',
                  'text-align': 'center'}),
    ], style={'margin-top': '5rem'}
)

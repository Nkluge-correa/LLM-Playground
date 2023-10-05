from dash import dcc, html, Output, Input, State, callback
import dash_bootstrap_components as dbc
from datetime import datetime
import pandas as pd
import random
import dash

from badges import badges
from toggle import toggle_modal, toggle_offcanvas
from llm_elements import query_models, modal_info, llm_textbox

dash.register_page(__name__,
                   path='/language-model-playground',
                   title='LLM Playground 🎮',
                   name='LLM Playground 🎮')

prompts = ["The capybara crossed the road because",
           "Machine learning is the study of",
           "The moon is made of",
           "The best way to learn is",]

layout = html.Div(
    children=[
        html.Div([dcc.Markdown('# LLM Playground', className='title-style', style={'color': '#dc3d87'}),
                  html.Img(src=dash.get_asset_url(
                      'logo.svg'), height="60px", className='title-icon-style')],
                 className='title-div'),
        html.Div([
            html.Div([
                dcc.Markdown('''
                The Large Language Model (LLM) Playground is a simple application that allows the user to interact and experiment with LLMs.
                             
                `(Powered by Hugging Face Inference API 🤗)`
                ''', className='page-intro')
            ], className='page-intro-inner-div'),
        ], className='page-intro-outer-div'),
        html.Div([modal_info],
                 className='middle-toggles'),
        dbc.Row([
            dbc.Col([
                html.Div(
                    [
                        dbc.Offcanvas(
                            dbc.InputGroup(
                                style={'width': '100%', 'max-width': '100vw',
                                       'margin': 'auto'},
                                children=[
                                    html.Div([
                                        html.H2(
                                            "Sampling Parameters ⚙️", className="nav-header"),
                                        html.Hr(),
                                        dbc.Label("Choose a language model",
                                                  className='modal-body-text-style', color='#dc3d87'),
                                        dbc.RadioItems(
                                            options=[
                                                {"label": "DistilGPT2",
                                                 "value": "models/distilgpt2"},
                                                {"label": "GPT-2 small",
                                                    "value": "models/gpt2"},
                                                {"label": "GPT-2 large",
                                                    "value": "models/gpt2-large"},    
                                                {"label": "BLOOM",
                                                    "value": "models/bigscience/bloom"},
                                                
                                            ],
                                            value="models/gpt2",
                                            id="choose-model-input",
                                            inline=True,
                                            style={'margin-bottom': '10px',
                                                   'display': 'flex', 'flex-direction': 'column'}
                                        ),
                                        dbc.Label("Top-K",
                                                  className='modal-body-text-style', color='#dc3d87',
                                                  style={'margin-top': '10px'}),
                                        dcc.Slider(10, 100, 10, value=30, marks={10: '10', 100: '100'},
                                                   tooltip={"placement": "bottom",
                                                            "always_visible": True},
                                                   id='topk-slider'),
                                        dbc.Label("Top-P",
                                                  className='modal-body-text-style', color='#dc3d87',
                                                  style={'margin-top': '10px'}),
                                        dcc.Slider(0.1, 0.95, 0.05, value=0.3, marks={0.1: '0.1', 1: '1'},
                                                   tooltip={"placement": "bottom",
                                                            "always_visible": True},
                                                   id='topp-slider'),
                                        dbc.Label("Temperature",
                                                  className='modal-body-text-style', color='#dc3d87',
                                                  style={'margin-top': '10px'}),
                                        dcc.Slider(0.1, 2.0, 0.1, marks={0.1: '0.1', 2: '2'}, value=0.5,
                                                   tooltip={'placement': 'bottom',
                                                            'always_visible': True},
                                                   id='temperature-slider'),
                                        dbc.Label("Response Length",
                                                  className='modal-body-text-style', color='#dc3d87',
                                                  style={'margin-top': '10px'}),
                                        dcc.Slider(25, 250, 25, value=100, marks={25: '25', 250: '250'},
                                                   tooltip={'placement': 'bottom',
                                                            'always_visible': True},
                                                   id='length-slider',),
                                        dbc.Label("Repetition Penalty",
                                                  className='modal-body-text-style', color='#dc3d87',
                                                  style={'margin-top': '10px'}),
                                        dcc.Slider(0.1, 2.0, 0.1, marks={0.1: "0.1", 2.1: "2.0"}, value=1.5,
                                                   tooltip={'placement': 'bottom',
                                                            'always_visible': True},
                                                   id='repetition-slider'),
                                    ], style={
                                        'width': '100%',
                                        'max-width': '100vw',
                                        'margin': 'auto',
                                        'background-color': 'none'})]),
                            id="off-sampling-llm",
                            title="",
                            is_open=False,
                            style={'width': '22rem'},
                            scrollable=True,
                            close_button=False,
                            placement='end',
                        ),
                    ]
                ),
                html.Div([dbc.InputGroup(
                    style={'width': '100%', 'max-width': '100vw',
                           'margin': 'auto'},
                    children=[
                        dcc.Textarea(
                            id='textarea-state',
                            value=random.choice(prompts),
                            title='''LLMs sometimes write plausible-sounding but incorrect or nonsensical answers. DO NOT believe everything you read. LLMs are sensitive to their inputs. Given a certain prompt, these models can generate toxic/harmful content. DO NOT use these models for high-stakes applications without the proper safety precautions. LLMs can be repetitive and verbose.''',
                            style={'width': '100%', 'height': '50px',
                                   'color': '#313638', 'background-color': '#f2f2f2'},
                        ),
                        html.Div([
                            dbc.Button([html.I(className="bi bi-download")],
                                       id='download-button', disabled=True,
                                       outline=True, color="light", style={
                                'border-radius': '5px',
                                'margin': '2px',
                                'width': '25%',
                                'max-width': '100vw',
                                'background-color': 'none'}),
                            dbc.Button(
                                [html.I(className="bi bi-send")], size='lg', id='submit-button',
                                title='''LLMs sometimes write plausible-sounding but incorrect or nonsensical answers. DO NOT believe everything you read. LLMs are sensitive to their inputs. Given a certain prompt, these models can generate toxic/harmful content. DO NOT use these models for high-stakes applications without the proper safety precautions. LLMs can be repetitive and verbose.''',
                                outline=True, color='light', style={
                                    'border-radius': '5px',
                                    'margin': '2px',
                                    'width': '50%',
                                    'max-width': '100vw',
                                    'background-color': 'none'}),
                            dbc.Button(
                                [html.I(className="bi bi-gear")], size='lg', id='open-sampling-llm',
                                outline=True, color='light', style={
                                    'border-radius': '5px',
                                    'margin': '2px',
                                    'width': '25%',
                                    'max-width': '100vw',
                                    'background-color': 'none'}),
                        ], style={'display': 'flex', 'flex-direction': 'row',
                                  'width': '100%', 'max-width': '100vw'}),
                    ],
                )], style={
                    'margin-bottom': '20px', 'margin-top': '25px'}),
                dcc.Loading(id='loading_0', type='circle', color='#dc3d87', children=[html.Div(id='textarea-output-state',
                                                                                               contentEditable="true",
                                                                                               style={'width': '100%',
                                                                                                      'max-width': '100vw',
                                                                                                      'min-height': '70vh',
                                                                                                      'max-height': '2024px',
                                                                                                      'margin': 'auto',
                                                                                                      "overflow": "auto",
                                                                                                      "display": "flex",
                                                                                                      "flex-direction": "column-reverse",
                                                                                                      'margin-top': '25px',
                                                                                                      'outline': 'white solid 1px',
                                                                                                      })]),
                dcc.Download(id="download-session"),
                dcc.Store(id='store-session', data=''),
            ], md=12),
        ]),
        html.Div([
            html.Div([badges], className='badges'),
        ], className='badges-div'),
    ],
)


@callback(
    Output("off-sampling-llm", "is_open"),
    Input("open-sampling-llm", "n_clicks"),
    [State("off-sampling-llm", "is_open")],
    suppress_callback_exceptions=True,
    prevent_initial_call=True,
)
def toggle_llm_sampling(n1, is_open):
    return toggle_offcanvas(n1, is_open)


@callback(

    [Output('textarea-output-state', 'children'),
     Output('store-session', 'data'),
     Output('download-button', 'disabled'),
     Output('download-button', 'outline')],

    Input('submit-button', 'n_clicks_timestamp'),

    [State('textarea-state', 'value'),
     State('length-slider', 'value'),
     State('temperature-slider', 'value'),
     State('topk-slider', 'value'),
     State('topp-slider', 'value'),
     State('repetition-slider', 'value'),
     State('choose-model-input', 'value'),
     State('store-session', 'data')],
    prevent_initial_call=True,
)
def update_output(
        click, string,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        api_end_point,
        session_history):
    """
    Update the output of the app with a generated text and update the session history.

    Parameters:
    -----------
        click : int or None
            Number of times the input button has been clicked.
        string : str
            User input string.
        max_new_tokens : int
            Maximum number of tokens to generate.
        temperature : float
            Controls the "creativity" of the generated text.
        top_k : int
            Controls the diversity of the generated text.
        repetition_penalty : float
            Controls the level of repetition in the generated text.
        api_end_point : str
            API endpoint for the GPT model.
        session_history : list of lists, optional
            List of lists with the following format:
            [timestamps, models, temperatures, top_ks, repetition_penalties, prompts, generated_responses],
            where each list corresponds to a column of the dataframe containing the session history.

    Returns:
    --------
        text_output : str or list
            A string with an error message if the user did not input anything or a list
            with a markdown header and the generated text.
        session_history : list of lists
            Updated list of lists containing the session history.
        clear_input : bool
            Whether or not to clear the input field.
    """

    session_history = session_history or [[], [], [], [], [], [], [], []]

    if click is not None:

        if string.isspace() is True:
            raise dash.exceptions.PreventUpdate

        if not string:
            raise dash.exceptions.PreventUpdate

        if string:

            generated_text, model = query_models(api_end_point, string, max_new_tokens,
                                                 temperature,
                                                 top_k, top_p, repetition_penalty)

            session_history[7].insert(0, (generated_text, model))

            text_output = [llm_textbox(text, model)
                           for text, model in session_history[7]]

            now = datetime.now()
            now_string = now.strftime("%H:%M:%S-%d/%m/%Y")

            session_history[0].append(now_string)
            session_history[1].append(model)
            session_history[2].append(temperature)
            session_history[3].append(top_k)
            session_history[4].append(repetition_penalty)
            session_history[5].append(string)
            session_history[6].append(generated_text)

            return text_output, session_history, False, False
    else:

        return '', session_history, True, True


@callback(
    Output('modal-info-llm', 'is_open'),
    [
        Input('info-llm-button', 'n_clicks'),
        Input('close-info-llm', 'n_clicks'),
    ],
    [State('modal-info-llm', 'is_open')],
)
def toggle_info_llm(n1, n2, is_open):
    return toggle_modal(n1, n2, is_open)


@callback(
    Output("download-session", "data"),
    Input("download-button", "n_clicks"),
    State('store-session', 'data'),
    prevent_initial_call=True,
)
def download_session(n_clicks, session_history):
    """
    This function takes two arguments, n_clicks and session_history, and 
    returns a CSV file that contains a dataframe built from the session_history data.

    If session_history is not provided, an empty list is assigned to it. 
    The function then creates a Pandas dataframe from the session_history list, 
    with column names for different parts of the history (model, temperature, 
    top_k, repetition_penalty, prompt, and generated_response) and index set to 
    the first element of session_history. The index is then named date, and the 
    dataframe is saved to a CSV file in the data directory using the to_csv method. 
    Finally, the function returns a Dash dcc.send_file object with the CSV file, 
    which allows the user to download the file.

    Overall, this function seems to be designed to allow a user to download a 
    history of their interactions with a language model, including the prompts 
    they entered and the responses generated by the model. However, it assumes 
    that session_history is a list with a specific format and that the user has 
    already interacted with the language model and recorded this history somewhere. 
    Without more context about how this function is used in a larger application or 
    workflow, it is difficult to say more about its purpose.
    """
    session_history = session_history or []

    df = pd.DataFrame({
        "model": session_history[1],
        "temperature": session_history[2],
        "top_k": session_history[3],
        "repetition_penalty": session_history[4],
        "prompt":  session_history[5],
        "generated_response":  session_history[6],
    }, index=session_history[0])

    df.index.name = 'date'
    df.to_csv('data/language_model_playground_session.csv')

    return dcc.send_file('data/language_model_playground_session.csv')

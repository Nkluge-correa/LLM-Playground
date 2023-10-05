import dash_bootstrap_components as dbc
from dash import dcc, html
import requests
import os

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

with open('config.env') as f:
    os.environ.update(
        line.replace('export ', '', 1).strip().split('=', 1) for line in f
        if 'export' in line
    )


def llm_textbox(text, model):
    style = {'max-width': '100%',
             'width': 'max-content',
             'padding': '10px 10px',
             'margin-bottom': '20px',
             'backdrop-filter': 'blur(14px)',
             "border-radius": '10px 10px 10px 10px',
             "box-shadow": '5px 10px 8px #1c1c1b',
             }

    style_text = {'text-align': 'justify',
                  'text-justify': 'inter-word'}

    return dbc.Card([dbc.CardHeader(dcc.Markdown(f'''## Generated Response (`{model}`)''')),
                     dbc.CardBody([dcc.Markdown(text, style=style_text),]),],
                    color="#272b30", outline=True, style=style)


def query_models(api_end_point: str, string: str, max_new_tokens: int,
                 temperature: float, top_k: int, top_p: float, repetition_penalty: float):
    """
    Queries the specified Hugging Face model API endpoint with the 
    given string input, requesting the generation of a new text sequence 
    of up to max_new_tokens length.

    Parameters:
    -----------
        - api_end_point (str): the API endpoint of the Hugging Face model to be queried
        - string (str): the input text string to be used as the basis for the generated text
        - max_new_tokens (int): the maximum number of new tokens to be generated
        - temperature (float): the sampling temperature to be used in generating the new text (higher values lead to more random output)
        - top_k (int): the number of top-k candidates to be considered at each decoding step
        - repetition_penalty (float): the penalty to be applied to repeated tokens in the generated text (higher values lead to less repetition)

    Returns:
    ----------
        - A tuple containing:
            1. A string representing the generated text from the queried model
            2. A string representing the name of the queried model
    """

    payload = {"inputs": string,
            "parameters": {"top_k": top_k, "top_p": top_p, "temperature": temperature,
                            "max_new_tokens": max_new_tokens,
                            "num_return_sequences": 1,
                            "repetition_penalty": repetition_penalty,
                            'max_time': 30,
                            'do_sample': True,
                            'return_full_text': True},
            "options": {'use_cache': False,
                        'wait_for_model': True}}
    try:

        response = requests.post(
            f"https://api-inference.huggingface.co/{api_end_point}",
            headers={"Authorization": f"Bearer {os.environ.get('API_KEY')}"},
            json=payload).json()

        logger.info(f"{api_end_point.split('/')[-1].title()} online")
        logger.info(response)

        return response[0]['generated_text'], api_end_point.split('/')[-1].title()

    except:

        logger.info(f"{api_end_point.split('/')[-1].title()} offline")

        return f"Server error. {api_end_point.split('/')[-1].title()} is offline.", api_end_point.split('/')[-1].title()


modal_info = html.Div(
    [
        html.A([html.I(className='bi bi-info-circle')],
               id="info-llm-button", n_clicks=0, className="icon-button", style={'font-size': 25}),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle(dcc.Markdown(
                    '# Language Model Playground 🤖', style={'font-weight': 'bold', 'color': '#dc3d87'}))),
                dbc.ModalBody([dcc.Markdown('''Since the transformer architecture was proposed by _Vaswani et al._ (Google) in their seminal paper "_[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)_," deep learning applied to NLP tasks has entered a new era: _the era of large language models_ (LLMs). Models such as [`BERT`](https://huggingface.co/docs/transformers/model_doc/bert), [`GPT-3`](https://arxiv.org/abs/2005.14165), [`LaMDA`](https://arxiv.org/abs/2201.08239), and [PaLM](https://arxiv.org/abs/2204.02311), are examples of LLM capable of solving many kinds of tasks.''', className='modal-body-text-style', ),
                               dcc.Markdown(
                                   '''However, when use as generative models, such systems can provide false, toxic, or simply useless content for their controller. Following in the line of research from similar works (Kenton et al. [2021](https://arxiv.org/pdf/2103.14659.pdf), Ziegler et al. [2022](https://arxiv.org/pdf/2205.01663.pdf), Ouyang et al. [2022](https://arxiv.org/pdf/2203.02155.pdf), Romal et al. [2022](https://arxiv.org/abs/2201.08239)), this research tool seeks to evaluate and improve the _alignment_ of language models, i.e., _how well the responses of such models are aligned with the intentions of a human controller_.''', className='modal-body-text-style', ),
                               dcc.Markdown(
                                   '''In the quest to create models that are better _[aligned](https://arxiv.org/abs/1906.01820)_ with human intentions, we hope to be able to create safer and more efficient models. In this playground, you can submit prompts to several language models and then compare their results.''', className='modal-body-text-style', ),
                               dcc.Markdown(
                                   '''For example, you can give a demonstration in the form of a prompt, and ask for a completion for an uncompleted task ([prompt engenniring/prompt tuning](https://en.wikipedia.org/wiki/Prompt_engineering)):''', className='modal-body-text-style', ),
                               dcc.Clipboard(target_id="text_recepy",
                                             style={"fontSize": 20}),
                               dcc.Markdown("""

                               ```markdown
                               
                               Recipe for chocolate cake that does not use eggs and milk:

                               - 1 cup of all-purpose flour, 1/2 cup of cocoa powder, 
                               1/2 teaspoon of baking soda, 1/2 teaspoon of baking powder, 
                               1/4 teaspoon of salt, 1/2 cup of sugar, 1/4 cup of vegetable oil, 
                               1/4 cup of water, 1 teaspoon of vanilla extract.

                               Recipe for a strawberry cake that does not use eggs and milk:

                               ```
                               """, id="text_recepy", className='modal-body-text-style'),
                               dcc.Markdown(
                                   '''An aligned language model would produce a cake recipe (_even if not very appetizing_) without using eggs and milk.''', className='modal-body-text-style', ),
                               dcc.Markdown(
                                   '''One possible strategy to promote alignment involves collecting examples of generated text for evaluation and correction purposes. Later, these processed samples can be used for fine-tuning. If you are interested in ML engineering and alignment, two of the most common techniques for working with this with LLM are _[Reinforcement Learning through Human Feedback](https://huggingface.co/blog/rlhf)_ and general _[fine-tuning](https://en.wikipedia.org/wiki/Fine-tuning)_ techniques.''', className='modal-body-text-style', ),
                               dcc.Clipboard(target_id="text_instruction",
                                             style={"fontSize": 20}),
                               dcc.Markdown("""

                               ```markdown

                               I hated this movie. Sentiment: Negative.
                               This movie is Ok. Sentiment: Neutral.
                               I loved this movie. Sentiment:

                               ```
                               """, id="text_instruction", className='modal-body-text-style', ),
                               dcc.Markdown('''_Note:_ The performance of the models may vary depending on the prompt. We recommend making it very clear what are the tasks to be completed (_verbosity usually helps in prompt engineering_). Also, it is useful to provide examples when asking for complicated tasks. Below we see an example of a prompt intended to cause an "_assistant_" type behavior:''',
                                            className='modal-body-text-style', ),
                               dcc.Clipboard(target_id="text_assistant",
                                             style={"fontSize": 20}),
                               dcc.Markdown('''
                               ```markdown
                                The conversations between a user and an AI assistant are shown below. The AI assistant makes an effort to be kind, considerate, honest, sophisticated, sensitive, and modest but knowledgeable. The assistant will try their best to comprehend what is required and is happy to assist with almost anything. Additionally, it makes an effort to avoid providing inaccurate or misleading information and warns when it is unsure of the correct response. Nevertheless, the assistant is practical, does its best, and avoids letting caution get in the way of being helpful.

                                ---

                                Human: What are the challenges posed by the alignment problem?

                                Assistant: The challenge of alignment is composed of two subproblems: outer alignment, which is the issue of aligning the true objectives of controllers to be optimized, and inner alignment, which involves aligning the base optimizer's objective with the Mesa objective of the model. This poses an ethical and philosophical problem of how to impart human values to machine learning models. Refer to "Risks from Learned Optimization in Advanced Machine Learning Systems" for further elaboration.

                                ---

                                Human: How do I open a CSV file in Python:

                                Assistant: You can use the `pandas` library in the following way:

                                import pandas as pd

                                df = pd.read_csv("your_file.csv")

                                ---

                                Human: What is stochastic gradient descent?

                                Assistant:

                               ```
                               ''', id="text_assistant", className='modal-body-text-style', ), html.Br(),
                               dcc.Markdown(
                                   '''## Working with Sampling Parameters''', style={'font-weight': 'bold', 'color': '#dc3d87'}), html.Br(),
                               dcc.Markdown('''
                               Language models usually generate text through _greedy search_, i.e., selecting the highest probability token at each autoregressive step. However, human language is not the output of a policy consecutively iterates a greedy policy. That's where sampling parameters come to aid us. They are the dials and knobs on the left portion of this page.''', className='modal-body-text-style', ),
                               dcc.Markdown('''With the proper sampling strategy, every token with a non-zero probability has a chance of being selected, and the different sampling parameters tweak the output generated by a model, e.g., by increasing the temperature parameter, which increases the entropy of the resulting softmax output, we can get a more diverse output.''',
                                            className='modal-body-text-style', ),
                               dcc.Markdown(
                                   '''Let's have a quick and holist review of what every parameter on the left panel controls (with the `Download` button, you can download your `prompt + generated response` history along with the selected parameters):''', className='modal-body-text-style', ),
                               dcc.Markdown(
                                   '''
                                   - **`Top-K`**: Controls the number of highest probability tokens to consider for each step.
                                   - **`Top-P`**: Controls the cumulative probability of the generated tokens.
                                   - **`Temperature`**: Controls the randomness of the generated tokens.
                                   - **`Response Length`**: Controls the maximum length of the generated text.                                                                                                 
                                   - **`Repetition penalty`**: Higher values help the model to avoid repetition in text generation.
                                   ''', className='modal-body-text-style', ), html.Br(),
                               dcc.Markdown('''## Models''', style={'font-weight': 'bold', 'color': '#dc3d87'}), html.Br(),
                               dcc.Markdown(
                                   '''This playground allows you to interact with four different models:''', className='modal-body-text-style', ),
                               dcc.Markdown(
                                   '''
                                   - [DistilGPT2](https://huggingface.co/distilgpt2): `DistilGPT2` (short for Distilled-GPT2) is an English-language model pre-trained with the supervision of the smallest version of [Generative Pre-trained Transformer 2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (`GPT-2`). `DistilGPT2` has 82 million parameters and was developed by [Hugging Face](https://huggingface.co/).
                                   - [GPT-2 small](https://huggingface.co/gpt2): `GPT-2` is a large language model pre-trained on a very large corpus of English data in a self-supervised fashion. The model offered in this end-point is the smallest version of `GPT-2` (124M). `GPT-2` was developed by [OpenAI](https://openai.com/).
                                   - [GPT-2 large](https://huggingface.co/gpt2): `GPT-2` is a large language model pre-trained on a very large corpus of English data in a self-supervised fashion. The model offered in this end-point is the large version of `GPT-2` (774M). `GPT-2` was developed by [OpenAI](https://openai.com/).
                                   - [BLOOM](https://huggingface.co/bigscience/bloom):  `BLOOM`  is an autoregressive large language model trained on vast amounts of text data using industrial-scale computational resources. It can output coherent text in 46 languages and 13 programming languages.  `BLOOM`  has more than 176B parameters and is one of the largest Open-access Multilingual Language models available, made possible by  [BigScience](https://bigscience.huggingface.co/).
                                   
                                   ''', className='modal-body-text-style', ),
                               dcc.Markdown(
                                   '''For more details, follow the links to read the `model card` of each model.''', className='modal-body-text-style', ), html.Br(),
                               dcc.Markdown(
                                   '''## Limitations and Risks''', style={'font-weight': 'bold', 'color': '#dc3d87'}), html.Br(),
                               dcc.Markdown(
                                   '''Large language models can produce toxic, discriminatory, and harmful language. They can also generate misinformation and simulate agency (users may attribute human characteristics to the model). Given the size and amount of data needed to train large language models, such systems also have a high environmental impact during their training.''', className='modal-body-text-style', ),
                               dcc.Markdown('''
                                - 🤥 Generative models can perpetuate the generation of pseudo-informative content, that is, false information that may appear truthful. For example, multi-modal generative models can be used to create images with untruthful content, while language models for text generation can automate the generation of misinformation.  
  
                                - 🤬 In certain types of tasks, generative models can generate toxic and discriminatory content inspired by historical stereotypes against sensitive attributes (for example, gender, race, religion). Unfiltered public datasets may also contain inappropriate content, such as pornography, racist images, and social stereotypes, which can contribute to unethical biases in generative models. Furthermore, when prompted with non-English languages, some generative models may perform poorly.  
                                
                                - 🎣 Generative models with high performance in conversational tasks can be used by malicious actors to intentionally cause harm through social engineering techniques like phishing and large-scale fraud. Also, anthropomorphizing AI models can lead to unrealistic expectations and a lack of understanding of the limitations and capabilities of the technology, which can result in potentially harmful decisions being made based on this misinformation.

                                - 🏭 The development of large machine learning models can have significant environmental impacts due to the high energy consumption required for their training. If the energy used to power their training process comes from burning fossil fuels, it can contribute to the injection of large amounts of CO2 into the atmosphere. Hyperparameter optimization (often necessary before final training) can also contribute to these energy-intensive tasks.

                                ''', className='modal-body-text-style', ),
                               dcc.Markdown(
                                   '''If you want to know more abut the risks associated with current ML models, visit our [Model Library](https://playground.airespucrs.org/model-library).''', className='modal-body-text-style', ), html.Br(),
                               dcc.Markdown(
                                   '''## Potential Uses & Out-of-scope Uses''', style={'font-weight': 'bold', 'color': '#dc3d87'}), html.Br(),
                               dcc.Markdown('''
                                   These models were created to enable public research on large language models. LLMs are intended to be used for language generation or as a pre-trained base model that can be further fine-tuned for specific tasks. Use cases for such models include: "_text generation, exploring characteristics of language generated by a language model, information extraction, question answering, summarization_".''', className='modal-body-text-style', ),
                               dcc.Markdown(
                                   '''Intended users are the "_general public, researchers, students, educators, engineers/developers, non-commercial entities, community advocates, including human and civil rights groups_."''', className='modal-body-text-style', ),
                               dcc.Markdown(
                                   '''Using any of these models in high-stakes settings is out of the scope of their intended use. These models were not designed for critical decisions nor use with any material consequences on an individual's livelihood or well-being. Generated content may appear factual but may not be correct. Out-of-scope uses include "_usage in biomedical domains, political and legal domains, or finance domains; usage for evaluating or scoring individuals, such as for employment, education, or credit; applying the model for critical automatic decisions, generating factual content, creating reliable summaries, or generating predictions that must be correct._"''', className='modal-body-text-style', ),
                               dcc.Markdown(
                                   '''Intentionally using the model for harm, violating human rights, or other kinds of malicious activities is a misuse of these models. This includes "_spam generation, misinformation and influence operations, disparagement and defamation, harassment and abuse, deception, unconsented impersonation and imitation, unconsented surveillance, generating content without attribution to the model,_" among other unlawful behaviors.''', className='modal-body-text-style', ), html.Br(),
                               dcc.Markdown(
                                   '''## Recommendations''', style={'font-weight': 'bold', 'color': '#dc3d87'}), html.Br(),
                               dcc.Markdown('''
                                    - LLMs sometimes write plausible-sounding but incorrect or nonsensical answers. DO NOT believe everything you read.

                                    - LLMs are sensitive to their inputs. Given a certain prompt, these models can generate toxic/harmful content. DO NOT use these models for high-stakes applications without the proper safety precautions.

                                    - LLMs can be repetitive and verbose.
                                   ''', className='modal-body-text-style', ),
                               ]),
                dbc.ModalFooter(
                    dbc.Button(
                        html.I(className="bi bi-x-circle-fill"),
                        id='close-info-llm',
                        className='ms-auto',
                        outline=True,
                        size='xl',
                        n_clicks=0,
                        color='primary',
                        style={'border': 0, 'font-weight': 'bold'}
                    )
                ),
            ],
            id='modal-info-llm',
            scrollable=True,
            fullscreen=True,
            is_open=False,
        ),
    ], style={
        'margin-top': '5px',
        'margin-right': '15px',
        "display": "inline-block",
    },

)

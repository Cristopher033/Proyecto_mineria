import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ==========================
#   CARGAR DATOS CSV
# ==========================
try:
    df = pd.read_csv("dataset_stack_limpio.csv")
    print(f"Dataset cargado exitosamente: {len(df)} registros")
    df = df.astype(object).fillna("No especificado")
except FileNotFoundError:
    print("Advertencia: No se encuentra 'dataset_stack_limpio.csv'")
    print("El dashboard de análisis no estará disponible.")
    df = None

# ==========================
#   MODELO PREDICTIVO
# ==========================
np.random.seed(42)
lenguajes_lista = [
    'Python', 'Java', 'C++', 'JavaScript', 'C#', 'PHP',
    'TypeScript', 'Go', 'Ruby', 'Swift', 'Kotlin', 'Rust'
]

data_modelo = pd.DataFrame({
    'Anios_Experiencia': np.random.randint(0, 15, 3000),
    'Nivel': np.random.choice(['Senior', 'Semi-Senior', 'Junior'], 3000),
    'Lenguaje': np.random.choice(lenguajes_lista, 3000),
    'Salario': np.random.randint(8000, 60000, 3000)
})

data_encoded = pd.get_dummies(data_modelo, columns=['Nivel', 'Lenguaje'], drop_first=True)
X = data_encoded.drop(columns='Salario')
y = data_encoded['Salario']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = LinearRegression()
modelo.fit(X_train, y_train)
score = modelo.score(X_test, y_test)

# ==========================
#   CONFIGURACIÓN APP
# ==========================
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True
)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body {
                background-color: #1a2233 !important;
                margin: 0;
                padding: 0;
            }
            .container-fluid {
                padding-top: 0 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.title = "Dashboard Stack Overflow"

# ==========================
#   PALETA DE COLORES
# ==========================
# Fondo principal: azul marino oscuro
BG_MAIN      = "#1a2233"
# Superficies de tarjetas
BG_CARD      = "#232f45"
BG_CARD2     = "#1e2a3e"
# Acento principal (azul pálido académico)
ACCENT       = "#4a90c4"
ACCENT_LIGHT = "#6aabe0"
# Texto
TEXT_PRIMARY  = "#dde3ec"
TEXT_MUTED    = "#8a97b0"
# Borde sutil
BORDER_COLOR = "#2e3d58"

# ==========================
#   ESTILOS
# ==========================
TAB_STYLE = {
    'borderBottom': f'1px solid {BORDER_COLOR}',
    'padding': '12px 20px',
    'fontWeight': '500',
    'borderRadius': '6px 6px 0 0',
    'background': BG_CARD2,
    'color': TEXT_MUTED,
    'fontSize': '14px'
}

TAB_SELECTED_STYLE = {
    'borderTop': f'2px solid {ACCENT}',
    'borderBottom': f'1px solid {BG_MAIN}',
    'padding': '12px 20px',
    'fontWeight': '600',
    'borderRadius': '6px 6px 0 0',
    'background': BG_CARD,
    'color': TEXT_PRIMARY,
    'fontSize': '14px'
}

CARD_STYLE = {
    'background': BG_CARD,
    'borderRadius': '8px',
    'border': f'1px solid {BORDER_COLOR}',
    'padding': '20px',
    'marginBottom': '20px'
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=BG_CARD2,
    font=dict(color=TEXT_PRIMARY, family="Arial, sans-serif", size=12),
    height=420,
    margin=dict(l=150, r=20, t=60, b=50),
    title_font=dict(size=15, color=TEXT_PRIMARY),
    xaxis=dict(
        gridcolor=BORDER_COLOR,
        linecolor=BORDER_COLOR,
        tickfont=dict(color=TEXT_MUTED)
    ),
    yaxis=dict(
        gridcolor=BORDER_COLOR,
        linecolor=BORDER_COLOR,
        tickfont=dict(color=TEXT_MUTED)
    )
)

# Secuencia de colores sobrios para gráficos
COLORSCALE_BARS = [
    [0.0,  "#2a4a7f"],
    [0.33, "#3a6ea8"],
    [0.66, "#4a90c4"],
    [1.0,  "#6aabe0"],
]

# ==========================
#   LAYOUT PRINCIPAL
# ==========================
app.layout = dbc.Container([

    # Header
    html.Div([
        html.H1(
            "Dashboard — Stack Overflow Survey",
            className="text-center mt-4 mb-1",
            style={'color': TEXT_PRIMARY, 'fontWeight': '600', 'fontSize': '1.6rem', 'letterSpacing': '0.5px'}
        ),
        html.P(
            "Análisis exploratorio de datos y modelo predictivo de salario",
            className="text-center mb-4",
            style={'color': TEXT_MUTED, 'fontSize': '0.95rem'}
        ),
        html.Hr(style={'borderColor': BORDER_COLOR, 'marginBottom': '0'})
    ]),

    # Tabs
    dcc.Tabs(
        id="tabs-principal",
        value='tab-analisis',
        children=[
            dcc.Tab(
                label='Análisis de Datos',
                value='tab-analisis',
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE,
                disabled=df is None
            ),
            dcc.Tab(
                label='Modelo Predictivo',
                value='tab-predictivo',
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE
            ),
        ],
        style={'marginBottom': '0', 'borderBottom': f'1px solid {BORDER_COLOR}'},
        colors={"border": BORDER_COLOR, "primary": ACCENT, "background": BG_CARD2}
    ),

    # Contenido dinámico
    html.Div(
        id='tabs-content',
        style={'paddingTop': '24px'}
    ),

    # Footer
    html.Footer([
        html.Hr(style={'borderColor': BORDER_COLOR}),
        html.P(
            "Mineria de datos",
            className="text-center mb-3",
            style={'color': TEXT_MUTED, 'fontSize': '0.85rem'}
        )
    ], className="mt-4"),

], fluid=True, style={
    'background': BG_MAIN,
    'minHeight': '100vh',
    'padding': '0 24px 24px 24px'
})

# ==========================
#   LAYOUT TAB ANÁLISIS
# ==========================
def crear_tab_analisis():
    if df is None:
        return dbc.Alert(
            "No se pudo cargar 'dataset_stack_limpio.csv'. Verifica que el archivo exista en el directorio.",
            color="warning",
            className="text-center"
        )

    return html.Div([
        # Tarjetas de estadísticas
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.P("Total desarrolladores", style={'color': TEXT_MUTED, 'fontSize': '0.8rem', 'marginBottom': '4px'}),
                        html.H4(f"{len(df):,}", style={'color': ACCENT_LIGHT, 'fontWeight': '600', 'marginBottom': '0'})
                    ])
                ], style=CARD_STYLE)
            ], width=12, md=3, className="mb-3"),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.P("Países representados", style={'color': TEXT_MUTED, 'fontSize': '0.8rem', 'marginBottom': '4px'}),
                        html.H4(f"{df['Country'].nunique()}", style={'color': ACCENT_LIGHT, 'fontWeight': '600', 'marginBottom': '0'})
                    ])
                ], style=CARD_STYLE)
            ], width=12, md=3, className="mb-3"),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.P("Tipos de desarrollador", style={'color': TEXT_MUTED, 'fontSize': '0.8rem', 'marginBottom': '4px'}),
                        html.H4(f"{df['DevType'].nunique()}", style={'color': ACCENT_LIGHT, 'fontWeight': '600', 'marginBottom': '0'})
                    ])
                ], style=CARD_STYLE)
            ], width=12, md=3, className="mb-3"),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.P("Registros con edad válida", style={'color': TEXT_MUTED, 'fontSize': '0.8rem', 'marginBottom': '4px'}),
                        html.H4(f"{(df['Age'] != 'No especificado').sum():,}", style={'color': ACCENT_LIGHT, 'fontWeight': '600', 'marginBottom': '0'})
                    ])
                ], style=CARD_STYLE)
            ], width=12, md=3, className="mb-3"),
        ]),

        # Filtros
        dbc.Card([
            dbc.CardBody([
                html.H6("Filtros", style={'color': TEXT_PRIMARY, 'fontWeight': '600', 'marginBottom': '16px'}),
                dbc.Row([
                    dbc.Col([
                        html.Label("País", style={'color': TEXT_MUTED, 'fontSize': '0.85rem', 'marginBottom': '6px'}),
                        dcc.Dropdown(
                            options=[{"label": c, "value": c} for c in sorted(df["Country"].unique())],
                            id="pais-filter",
                            value=None,
                            placeholder="Todos los países",
                            style={'borderRadius': '6px', 'fontSize': '13px'}
                        )
                    ], md=5),

                    dbc.Col([
                        html.Label("Tipo de desarrollador", style={'color': TEXT_MUTED, 'fontSize': '0.85rem', 'marginBottom': '6px'}),
                        dcc.Dropdown(
                            options=[{"label": c, "value": c} for c in sorted(df["DevType"].unique())],
                            id="tipo-filter",
                            value=None,
                            placeholder="Todos los tipos",
                            style={'borderRadius': '6px', 'fontSize': '13px'}
                        )
                    ], md=5),

                    dbc.Col([
                        html.Label("\u00a0", style={'color': TEXT_MUTED, 'fontSize': '0.85rem', 'marginBottom': '6px', 'display': 'block'}),
                        dbc.Button(
                            "Limpiar filtros",
                            id="btn-limpiar",
                            color="secondary",
                            outline=True,
                            className="w-100",
                            style={'borderRadius': '6px', 'fontSize': '13px', 'borderColor': BORDER_COLOR, 'color': TEXT_MUTED}
                        )
                    ], md=2),
                ]),
            ])
        ], style=CARD_STYLE),

        # Indicador de filtros activos
        html.Div(id="filtros-activos", className="mb-3"),

        # Gráficos fila 1
        dbc.Row([
            dbc.Col([dcc.Loading(type="circle", color=ACCENT, children=[dcc.Graph(id="grafico-lenguajes")])], md=6, className="mb-4"),
            dbc.Col([dcc.Loading(type="circle", color=ACCENT, children=[dcc.Graph(id="grafico-herramientas")])], md=6, className="mb-4"),
        ]),

        # Gráficos fila 2
        dbc.Row([
            dbc.Col([dcc.Loading(type="circle", color=ACCENT, children=[dcc.Graph(id="grafico-edad")])], md=6, className="mb-4"),
            dbc.Col([dcc.Loading(type="circle", color=ACCENT, children=[dcc.Graph(id="grafico-paises")])], md=6, className="mb-4"),
        ]),
    ])


# ==========================
#   LAYOUT TAB PREDICTIVO
# ==========================
def crear_tab_predictivo():
    return html.Div([
        dbc.Row([
            # Panel izquierdo: entradas
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H6("Perfil del desarrollador", style={'color': TEXT_PRIMARY, 'fontWeight': '600', 'marginBottom': '20px'}),

                        html.Label("Años de experiencia", style={'color': TEXT_MUTED, 'fontSize': '0.85rem', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Slider(
                            0, 15, 1,
                            value=3,
                            id="input-exp",
                            marks={i: {'label': str(i), 'style': {'color': TEXT_MUTED, 'fontSize': '11px'}} for i in range(0, 16, 3)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="mb-4"
                        ),

                        html.Label("Nivel profesional", style={'color': TEXT_MUTED, 'fontSize': '0.85rem', 'marginBottom': '6px', 'display': 'block'}),
                        dcc.Dropdown(
                            id="input-nivel",
                            options=[
                                {"label": "Junior", "value": 'Junior'},
                                {"label": "Semi-Senior", "value": 'Semi-Senior'},
                                {"label": "Senior", "value": 'Senior'}
                            ],
                            value='Junior',
                            style={'borderRadius': '6px', 'fontSize': '13px', 'marginBottom': '16px'}
                        ),

                        html.Label("Lenguaje principal", style={'color': TEXT_MUTED, 'fontSize': '0.85rem', 'marginBottom': '6px', 'display': 'block'}),
                        dcc.Dropdown(
                            id="input-lenguaje",
                            options=[{"label": l, "value": l} for l in lenguajes_lista],
                            value='Python',
                            style={'borderRadius': '6px', 'fontSize': '13px', 'marginBottom': '24px'}
                        ),

                        dbc.Button(
                            "Calcular predicción",
                            id="btn-predecir",
                            className="w-100",
                            style={
                                'background': ACCENT,
                                'border': 'none',
                                'borderRadius': '6px',
                                'fontWeight': '500',
                                'fontSize': '14px',
                                'padding': '10px'
                            }
                        )
                    ]),
                    style=CARD_STYLE
                )
            ], md=4),

            # Panel derecho: resultado
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H6("Resultado de la predicción", style={'color': TEXT_PRIMARY, 'fontWeight': '600', 'marginBottom': '16px'}),
                        dcc.Loading(
                            type="circle",
                            color=ACCENT,
                            children=[dcc.Graph(id="grafico-salario", config={'displayModeBar': False})]
                        ),
                        html.H3(
                            id="salario-texto",
                            className="text-center mt-2",
                            style={'color': ACCENT_LIGHT, 'fontWeight': '600', 'fontSize': '1.5rem'}
                        ),
                        html.Div(id="salario-detalle", className="text-center mt-1")
                    ]),
                    style=CARD_STYLE
                )
            ], md=8),
        ])
    ])


# ==========================
#   CALLBACK: CAMBIO DE TAB
# ==========================
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs-principal', 'value')
)
def render_content(tab):
    if tab == 'tab-analisis':
        return crear_tab_analisis()
    elif tab == 'tab-predictivo':
        return crear_tab_predictivo()


# ==========================
#   CALLBACKS: TAB ANÁLISIS
# ==========================
if df is not None:
    @app.callback(
        Output("grafico-lenguajes", "figure"),
        Output("grafico-herramientas", "figure"),
        Output("grafico-edad", "figure"),
        Output("grafico-paises", "figure"),
        Output("filtros-activos", "children"),
        Input("pais-filter", "value"),
        Input("tipo-filter", "value")
    )
    def actualizar_graficos(pais, tipo):
        df_filtrado = df.copy()
        filtros_texto = []

        if pais:
            df_filtrado = df_filtrado[df_filtrado["Country"] == pais]
            filtros_texto.append(f"País: {pais}")
        if tipo:
            df_filtrado = df_filtrado[df_filtrado["DevType"] == tipo]
            filtros_texto.append(f"Tipo: {tipo}")

        if filtros_texto:
            filtros_msg = dbc.Alert(
                f"Filtros activos: {' | '.join(filtros_texto)} — {len(df_filtrado):,} registros",
                color="info",
                className="text-center py-2",
                style={'fontSize': '13px', 'background': BG_CARD2, 'borderColor': ACCENT, 'color': TEXT_MUTED}
            )
        else:
            filtros_msg = html.Div()

        # --- Lenguajes ---
        lenguajes = df_filtrado["LanguageHaveWorkedWith"].dropna().astype(str).str.split(";").explode().value_counts().head(10)
        colors_len = [COLORSCALE_BARS[int(i / max(len(lenguajes)-1, 1) * 3)][1] for i in range(len(lenguajes))]
        fig_lenguajes = go.Figure(go.Bar(
            x=lenguajes.values, y=lenguajes.index, orientation='h',
            marker_color=colors_len,
            text=lenguajes.values, textposition='auto',
            textfont=dict(color=TEXT_PRIMARY, size=11)
        ))
        fig_lenguajes.update_layout(
            **PLOTLY_LAYOUT,
            title={'text': "Lenguajes más utilizados", 'x': 0.5, 'xanchor': 'center'},
        )

        # --- Herramientas ---
        herramientas = df_filtrado["ToolsTechHaveWorkedWith"].dropna().astype(str).str.split(";").explode().value_counts().head(10)
        colors_her = [COLORSCALE_BARS[int(i / max(len(herramientas)-1, 1) * 3)][1] for i in range(len(herramientas))]
        fig_herramientas = go.Figure(go.Bar(
            x=herramientas.values, y=herramientas.index, orientation='h',
            marker_color=colors_her,
            text=herramientas.values, textposition='auto',
            textfont=dict(color=TEXT_PRIMARY, size=11)
        ))
        fig_herramientas.update_layout(
            **PLOTLY_LAYOUT,
            title={'text': "Herramientas más utilizadas", 'x': 0.5, 'xanchor': 'center'},
        )

        # --- Edad ---
        df_edad = df_filtrado[df_filtrado["Age"] != "No especificado"]
        fig_edad = px.histogram(
            df_edad, x="Age", color="DevType",
            title="Distribución de edades por tipo",
            barmode="stack",
            color_discrete_sequence=["#2a4a7f", "#3a6ea8", "#4a90c4", "#6aabe0", "#8cc5f0"]
        )
        edad_layout = {**PLOTLY_LAYOUT}
        edad_layout['margin'] = dict(l=60, r=20, t=60, b=60)
        fig_edad.update_layout(**edad_layout, title={'x': 0.5, 'xanchor': 'center'})
        fig_edad.update_traces(marker_line_width=0)

        # --- Países ---
        paises = df_filtrado["Country"].value_counts().head(15)
        colors_pai = [COLORSCALE_BARS[int(i / max(len(paises)-1, 1) * 3)][1] for i in range(len(paises))]
        fig_paises = go.Figure(go.Bar(
            x=paises.values, y=paises.index, orientation='h',
            marker_color=colors_pai,
            text=paises.values, textposition='auto',
            textfont=dict(color=TEXT_PRIMARY, size=11)
        ))
        fig_paises.update_layout(
            **PLOTLY_LAYOUT,
            title={'text': "Distribución por país (top 15)", 'x': 0.5, 'xanchor': 'center'},
        )

        return fig_lenguajes, fig_herramientas, fig_edad, fig_paises, filtros_msg

    @app.callback(
        Output("pais-filter", "value"),
        Output("tipo-filter", "value"),
        Input("btn-limpiar", "n_clicks"),
        prevent_initial_call=True
    )
    def limpiar_filtros(n):
        return None, None


# ==========================
#   CALLBACKS: TAB PREDICTIVO
# ==========================
@app.callback(
    Output("grafico-salario", "figure"),
    Output("salario-texto", "children"),
    Output("salario-detalle", "children"),
    Input("btn-predecir", "n_clicks"),
    State("input-exp", "value"),
    State("input-nivel", "value"),
    State("input-lenguaje", "value"),
    prevent_initial_call=False
)
def predecir_salario(n, exp, nivel, lenguaje):
    if not n:
        fig = go.Figure()
        fig.add_annotation(
            text="Configura el perfil y presiona «Calcular predicción»",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=TEXT_MUTED)
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=300, xaxis=dict(visible=False), yaxis=dict(visible=False)
        )
        return fig, "", ""

    entrada = pd.DataFrame({
        'Anios_Experiencia': [exp],
        'Nivel': [nivel],
        'Lenguaje': [lenguaje]
    })
    entrada_encoded = pd.get_dummies(entrada, columns=['Nivel', 'Lenguaje'])
    entrada_encoded = entrada_encoded.reindex(columns=X.columns, fill_value=0)
    prediccion = modelo.predict(entrada_encoded)[0]

    # Gauge con colores sobrios
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediccion,
        delta={
            'reference': 25000,
            'increasing': {'color': ACCENT_LIGHT},
            'decreasing': {'color': "#c0392b"}
        },
        number={'prefix': "$", 'suffix': " MXN", 'font': {'size': 44, 'color': ACCENT_LIGHT}},
        title={'text': "Salario estimado (MXN)", 'font': {'size': 14, 'color': TEXT_MUTED}},
        gauge={
            'axis': {
                'range': [0, 60000],
                'tickwidth': 1,
                'tickcolor': BORDER_COLOR,
                'tickfont': {'color': TEXT_MUTED, 'size': 11}
            },
            'bar': {'color': ACCENT, 'thickness': 0.7},
            'bgcolor': BG_CARD2,
            'borderwidth': 1,
            'bordercolor': BORDER_COLOR,
            'steps': [
                {'range': [0, 20000],  'color': "#1e2a3e"},
                {'range': [20000, 35000], 'color': "#1e3050"},
                {'range': [35000, 60000], 'color': "#1e3a60"},
            ],
            'threshold': {
                'line': {'color': ACCENT_LIGHT, 'width': 2},
                'thickness': 0.75,
                'value': prediccion
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': TEXT_PRIMARY},
        height=360,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    texto = f"${prediccion:,.0f} MXN / mes"

    if prediccion < 20000:
        rango = "Rango inicial"
        color = TEXT_MUTED
    elif prediccion < 35000:
        rango = "Rango competitivo"
        color = ACCENT_LIGHT
    else:
        rango = "Rango senior"
        color = "#5dbb8f"

    detalle = html.Div([
        html.Span(rango, style={'color': color, 'fontWeight': '500', 'fontSize': '13px'}),
        html.Br(),
        html.Small(
            f"Perfil: {nivel} · {lenguaje} · {exp} años de experiencia",
            style={'color': TEXT_MUTED, 'fontSize': '12px'}
        )
    ])

    return fig, texto, detalle


# ==========================
#   EJECUCIÓN
# ==========================
if __name__ == "__main__":
    print("Iniciando Dashboard...")
    print("Abre tu navegador en: http://localhost:8050")
    if df is None:
        print("Solo la pestaña de Modelo Predictivo estará disponible.")
    app.run(debug=False, host='0.0.0.0', port=8050)

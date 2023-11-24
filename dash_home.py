# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ÁREA DE IMPORTAÇÃO DE BIBLIOTECAS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# stremlit e companhia
import streamlit as st
#import streamlit_option_menu as som
#from streamlit_option_menu import option_menu

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Manipulação de dados e arquivos
import pandas as pd
import glob
import os


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Visualização de dados
import matplotlib.pyplot as plt
import seaborn as sns

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Estatística
import numpy as np
import scipy.stats as stats

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Resultados das performances dos modelos para avaliar os que apresentam melhor
# desempenho
# Os resultados estão no arquivo resultados_performance_modelos_{cluster}.csv
        # Calcular os resultados da performance


def CalcularPerformanceModelo(cluster, modelo):

    # carregar o arquivo de performance dos modelos
    # carregar o arquivo de performance dos modelos

    # filtrar o modelo
    df = df[df['modelo'] == modelo]
    resume.qtd_estacoes = len(df)
    resume.qtd_r2_score_abaixo_zero = len(df[df['r2_score'] < 0])
    resume.qtd_r2_score_abaixo_50 = len(df[(df['r2_score'] >= 0) & (df['r2_score'] < 0.5)])
    resume.qtd_r2_score_abaixo_60 = len(df[(df['r2_score'] >= 0.5) & (df['r2_score'] < 0.6)])
    resume.qtd_r2_score_abaixo_70 = len(df[(df['r2_score'] >= 0.6) & (df['r2_score'] < 0.7)])
    resume.qtd_r2_score_abaixo_80 = len(df[(df['r2_score'] >= 0.7) & (df['r2_score'] < 0.8)])
    resume.qtd_r2_score_acima_80 = len(df[df['r2_score'] >= 0.8])    
    return resume

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Cálculo de intervalo de confiança com bootstrap
import numpy as np

def bootstrap_ci(data, n_bootstraps=1000, alpha=0.05):
    """
    Calculate confidence intervals for the mean and median of a dataset using bootstrap.
    
    Parameters:
    data (array-like): The data to calculate the confidence intervals for.
    n_bootstraps (int): The number of bootstrap samples to generate.
    alpha (float): The significance level of the confidence interval.
    
    Returns:
    A tuple containing the confidence intervals for the mean and median.
    """
    # Generate bootstrap samples
    bootstrap_samples = np.random.choice(data, size=(n_bootstraps, len(data)), replace=True)
    
    # Calculate the mean and median of each bootstrap sample
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    bootstrap_medians = np.median(bootstrap_samples, axis=1)
    
    # Calculate the confidence intervals for the mean and median
    mean_ci = np.percentile(bootstrap_means, q=[100*alpha/2, 100*(1-alpha/2)])
    median_ci = np.percentile(bootstrap_medians, q=[100*alpha/2, 100*(1-alpha/2)])
    
    return mean_ci, median_ci


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Classe para tratar os dados das tabelas resultados_ML*.csv
# 
class ResumeML:
    def __init__(self, df, codigo):
        self.codigo = None
        self.cobacia = None
        self.cocursodag = None
        self.q95 = None
        self.media = None
        self.media_lower_bound = None
        self.media_upper_bound = None
        self.mediana = None
        self.mediana_lower_bound = None
        self.mediana_upper_bound = None
        self.dist_lower_bound = None
        self.dist_upper_bound = None
        self.dentro_dist = None
        self.error_media = None
        self.error_mediana = None
        self.moda = None
    
    # Calcular os indicadores estatísticos
    def calc_indicadores(self, df, y_pred, codigo):
        # calcular os indicadores estatísticos
        # loop para montar a página de análise de dados agrupado por estação
        self.codigo = codigo
        self.cobacia = df['cobacia'].unique()[0]
        self.cocursodag = df['cocursodag'].unique()[0]
        self.q95 = df['q95'].unique()[0]
        self.media = df[y_pred].mean()
        self.mediana = df[y_pred].median()
        # calcular os limites de confiança 0,025 e 0,975
        mean_ci, median_ci = bootstrap_ci(df[y_pred])
        self.media_lower_bound = mean_ci[0]
        self.media_upper_bound = mean_ci[1]
        self.mediana_lower_bound = median_ci[0]
        self.mediana_upper_bound = median_ci[1] 
        # calcular os limites de confiança 0,025 e 0,975 da distribuição
        self.dist_lower_bound = np.quantile(df[y_pred], 0.025)
        self.dist_upper_bound = np.quantile(df[y_pred], 0.975)
        # verificar se a q95 está dentro do intervalo de confiança da distribuição
        self.dentro_dist = (self.dist_lower_bound <= self.q95) & (self.q95 <= self.dist_upper_bound)
        self.error_media = (self.media - self.q95)/self.q95
        self.error_mediana = (self.mediana - self.q95)/self.q95
        self.moda = df[y_pred].mode()[0]

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# definição de funções
def show_ml(my_ml, my_ml2, cluster, tipo_painal = 'Agregado por modelo'):
    df = pd.read_csv('resultados_ML_'+my_ml+'_'+cluster+'.csv')

    # Alterar o tipo das colunas para string
    df['codigo'] = df['codigo'].astype(str)
    df['cobacia'] = df['cobacia'].astype(str)
    df['cocursodag'] = df['cocursodag'].astype(str)

    
    resume_list = []
    for codigo in df['codigo'].unique():
        # montar a página de análise de dados
        resume = ResumeML(df[df['codigo'] == codigo], codigo)
        resume.calc_indicadores(df[df['codigo'] == codigo], my_ml2, codigo)
        resume_list.append(resume.__dict__)
    # criar dataframe a partir da lista de dicionários
    resultado_df = pd.DataFrame.from_dict(resume_list)

    # colunas com indicadores estatísticos
    col1, col2, col3 = st.columns(3)
    col1.metric('Qtd Estações', int(resultado_df['codigo'].count()))
    col2.metric('Qtd Dentro Dist', int(resultado_df['dentro_dist'].sum()))
    col3.metric('% Dentro Dist', float(resultado_df['dentro_dist'].sum()/resultado_df['codigo'].count()).__round__(4) * 100)

    # verificar o tipo de painel
    if tipo_painal == 'Agregado por modelo':
        with st.expander(f"⏰ Tabela com os dados do modelo {my_ml}"):
            showData=st.multiselect('Filter: ',resultado_df.columns.tolist(),default=resultado_df.columns.tolist(), key=my_ml)
            st.dataframe(resultado_df[showData],use_container_width=True, height=200)
    elif tipo_painal == 'Resultado Performance Modelo':
        with st.expander(f"Performance do modelo {my_ml}"):
            df = pd.read_csv('resultado_performance_modelos_'+cluster+'.csv')
            # filtrar o modelo
            df = df[df['modelo'] == my_ml]
            # criar 3 colunas
            col1, col2, col3 = st.columns(3)
            # Quantidade de modelos com r2_score < 0 do conteúdo do dataframe
            col1.metric('r2_score < 0', df[df['r2_score'] < 0].count()[0])
            # Quantidade de modelos com r2_score >= 0 e < 0.5
            col2.metric('0 <= r2_score < 0.5', df[(df['r2_score'] >= 0) & (df['r2_score'] < 0.5)].count()[0])
            # Quantidade de modelos com r2_score >= 0.5 e < 0.6
            col3.metric('0.5 <= r2_score < 0.6', df[(df['r2_score'] >= 0.5) & (df['r2_score'] < 0.6)].count()[0])
            # criar mais 3 colunas
            col4, col5, col6 = st.columns(3)
            # Quantidade de modelos com r2_score >= 0.6 e < 0.7
            col4.metric('0.6 <= r2_score < 0.7', df[(df['r2_score'] >= 0.6) & (df['r2_score'] < 0.7)].count()[0])
            # Quantidade de modelos com r2_score >= 0.7 e < 0.8
            col5.metric('0.7 <= r2_score < 0.8', df[(df['r2_score'] >= 0.7) & (df['r2_score'] < 0.8)].count()[0])
            # Quantidade de modelos com r2_score >= 0.8
            col6.metric('0.8 <= r2_score', df[df['r2_score'] >= 0.8].count()[0])

            # carregar planilha com os resultados de performance dos modelos
            showData=st.multiselect('Filter: ',df.columns.tolist(),default=df.columns.tolist(), key=my_ml)
            st.dataframe(df[showData],use_container_width=True, height=200)

    elif tipo_painal == 'Agregado por estação':
        with st.expander(f"⏰ Gráfico das estações do Modelo {my_ml}"):
            # lista de estaçoes para seleção e montagem de gráficos
            estacoes = resultado_df['codigo'].unique()
            estacoes.sort()
            estacao = st.selectbox('Selecione a estação', estacoes, key=my_ml + 'estacao')
            # montar histograma dos dados y_pred em duas colunas

            # q95
            st.metric('Q95', float(resultado_df[resultado_df['codigo'] == estacao]['q95'].values[0]).__round__(2))
            # quantidade de registros da estaçao
            st.metric('Qtd Registros', int(df[df['codigo'] == estacao][my_ml2].count()))
            # moda
            st.metric('Moda', float(resultado_df[resultado_df['codigo'] == estacao]['moda'].values[0]).__round__(2))

            kpi1, kpi2 = st.columns(2)
            # média no kpi1
            kpi1.metric('Média', float(resultado_df[resultado_df['codigo'] == estacao]['media'].values[0]).__round__(2))
            # lower bound no kpi1
            kpi1.metric('Lower Bound', float(resultado_df[resultado_df['codigo'] == estacao]['media_lower_bound'].values[0]).__round__(2))
            # upper bound no kpi1
            kpi1.metric('Upper Bound', float(resultado_df[resultado_df['codigo'] == estacao]['media_upper_bound'].values[0]).__round__(2))

            # mediana no kpi2
            kpi2.metric('Mediana', float(resultado_df[resultado_df['codigo'] == estacao]['mediana'].values[0]).__round__(2))
            # lower bound no kpi2
            kpi2.metric('Lower Bound', float(resultado_df[resultado_df['codigo'] == estacao]['mediana_lower_bound'].values[0]).__round__(2))
            # upper bound no kpi2
            kpi2.metric('Upper Bound', float(resultado_df[resultado_df['codigo'] == estacao]['mediana_upper_bound'].values[0]).__round__(2))
            


            col1, col2 = st.columns(2)

            with col1:
                # montar gráfico histograma com linha vertical na q95 vermelha e tracejada, linhas verdes nos limites de confiança
                # quantile(0.025) e quantile(0.975), linha azul na média
                fig, ax = plt.subplots()
                sns.histplot(data=df[df['codigo'] == estacao], x=my_ml2, bins=100, ax=ax)
                ax.axvline(x=resultado_df[resultado_df['codigo'] == estacao]['q95'].values[0], color='red', linestyle='--')
                ax.axvline(x=resultado_df[resultado_df['codigo'] == estacao]['media'].values[0], color='blue', linestyle='--')
                ax.axvline(x=resultado_df[resultado_df['codigo'] == estacao]['dist_lower_bound'].values[0], color='green', linestyle='--')
                ax.axvline(x=resultado_df[resultado_df['codigo'] == estacao]['dist_upper_bound'].values[0], color='green', linestyle='--')
                ax.set_title('Histograma dos dados preditos')
                ax.set_xlabel('Vazão (m3/s)')
                ax.set_ylabel('Frequência')
                st.pyplot(fig)
            with col2:
                # montar gráfico q-q plot
                fig, ax = plt.subplots()
                stats.probplot(df[df['codigo'] == estacao][my_ml2], dist='norm', plot=ax)
                ax.set_title('Q-Q plot dos dados preditos')
                ax.set_xlabel('Quantis teóricos')
                ax.set_ylabel('Quantis observados')
                st.pyplot(fig)
    elif tipo_painal == 'Correlação de Features':
        st.write("Carregou correlação de features")
        # carregar o arquivo em um dataframe
        df = pd.read_csv('correlacao_features.csv')
        # deletar colunas desnecessárias = codigo, cobacia, cocursodag
        df.drop(['codigo', 'cobacia', 'cocursodag'], axis=1, inplace=True)
        # deletar as linhas cujo valor da coluna Unmaded: 0 seja = codigo, cobacia, cocursodag
        df.drop(df[df['Unnamed: 0'] == 'codigo'].index, inplace=True)
        df.drop(df[df['Unnamed: 0'] == 'cobacia'].index, inplace=True)
        df.drop(df[df['Unnamed: 0'] == 'cocursodag'].index, inplace=True)
        df.rename(columns={"Unnamed: 0": "features"}, inplace=True)
        df.set_index("features", inplace=True)
        # deletar a coluna index
        


        st.write(df)
        
        # criar um gráfico de correlação
        fig, ax = plt.subplots()
        # determinar o tamanho da fonte
        sns.set(font_scale=0.5)
        sns.heatmap(df, annot=True, ax=ax, cmap='coolwarm')
        st.pyplot(fig)

    else:
        st.write('Tipo de painel não implementado')

   


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Dicionário com os modelos disponíveis e seus respectivos campos de predição no arquivo
# de resultados


model_y_pred = { "brr": 'y_pred_brr',
                 "knnr": 'y_pred_knnr',
                 "lr": 'y_pred_LinearRegression',
                 "rfr": 'y_pred_RandomForestRegressor',
                 "svr": 'y_pred_svr',
                 "xgbr": 'y_pred_XGBRegressor',
                 "xgbrf": 'y_pred_XGBRFRegressor'}

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
st.set_option('deprecation.showPyplotGlobalUse', False)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SideBar   
st.sidebar.title('Menu')
st.sidebar.subheader('Escolha as opções de análise')

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Selecionar o tipo de análise, agregado por modelo ou por estação
analise = st.sidebar.radio('Selecione o tipo de análise', ['Agregado por modelo', 'Agregado por estação', 'Resultado Performance Modelo', 'Correlação de Features'])

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Monta as opcões de cluster para o usuário, utilizando o arquivo de clusters
# que foi gerado no código de análise de dados e disponíveis neste diretório

# carregar lista de arquivos a serem trabalhados
arquivos = glob.glob('resultados_ML_*')

# Busca todos os arquivos que começam com "resultados_ML_" e terminam com ".csv", cujo cluster está imediatamente antes do .csv
# e monta uma lista com os nomes dos clusters

clusters = [cluster.split('.')[0].split('_')[-1] for cluster in arquivos]
clusters = list(set(clusters))
clusters.sort()

# Cria um menu dropdown para o usuário selecionar o cluster desejado
cluster = st.sidebar.selectbox('Selecione o cluster', clusters)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# monta a lista de opções de modelos disponíveis para o cluster selecionado
# Busca todos os arquivos que começam com "resultados_ML_" e terminam com ".csv", cujo cluster está imediatamente antes do .csv
# e monta uma lista com os nomes dos modelos

ml = [ml.split('_')[2].split('_')[-1] for ml in arquivos if cluster in ml]
ml = list(set(ml))
ml.sort()   

# Cria um menu dropdown para o usuário selecionar o modelo desejado
modelo = st.sidebar.multiselect('Selecione o(s) modelo(s)', ml, default=ml)

#st.write(modelo)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Página Home - Introdução - Falar sobre a regionalização de vazões na Bacia do Rio São Francisco
# este texto aperece quando a página home é selecionada
st.title('Bem-vindo ao DashBoard de Vazões do Rio São Francisco')
st.subheader('Este projeto foi desenvolvido por: ')
st.write('''
- [Luís Eduardo]''')

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Montar a página de análise de dados 
# Para cada modelo selecionado pelo usuário serão mostrados indicadores dos valores preditos comparado à q95
# e uma tabela com os valores preditos e os valores observados, utilizando objeto mito_table

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Montar a página de análise de dados
# Para cada modelo selecionado pelo usuário serão mostrados indicadores dos valores preditos comparado à q95

# loop para montar a página de análise de dados agrupado por estação
for ml in modelo:
    # linha de separação
    st.markdown('---')
    # montar a página de análise de dados
    st.subheader('Resultados do ML '+ml)
    # montar a página de análise de dados
    show_ml(ml, model_y_pred[ml], cluster, analise)


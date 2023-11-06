st.set_option('deprecation.showPyplotGlobalUse', False)
# 1. Importar as bibliotecas necessárias
#from prometheus_client import Metric, MetricsHandler
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use o backend não interativo Agg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Teste de normalidade de Shapiro-Wilk
from scipy.stats import shapiro

#from pkgs.utilidades import BootstrapConfidenceInterval

st.set_option('deprecation.showPyplotGlobalUse', False)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# seção de definição de funções e classes

# classe para tratar os dados dos resultados do ML
class DataLoader():
    def __init__(self):
        self.data = pd.DataFrame()
        self.file = None
    # set file
    def set_file(self, file):
        self.file = file
    # get file
    def get_file(self):
        return self.file
    # carregar dados de um arquivo CSV
    def load_csv(self):
        self.data = pd.read_csv(self.file)
    # carregar dados de um arquivo Excel
    def load_excel(self, excel_file):
        self.data = pd.read_excel(self.file)
    # data
    #def get_data(self):
    #    return self.data

class DataLoaderResultadosML(DataLoader):
    def __init__(self):
        super().__init__()
    # get station codes
    def get_station_codes(self):
        return sorted(self.data["index"].unique().tolist())
    # get filtered data
    def get_filtered_data(self, station_code):
        return self.data[self.data["index"] == station_code]
    # get all statistics
    '''Estatísticas para da coluna y_pred para todas as estações'''
    def get_all_statistics(self):
        # 
        return self.data.describe()
    

class DataLoaderEstacoes(DataLoader):
    def __init__(self):
        super().__init__()
    # get station codes
    def get_station_codes(self):
        return sorted(self.data["Unnamed: 0"].unique().tolist())
    # get filtered data
    def get_filtered_data(self, station_code):
        return self.data[self.data["Unnamed: 0"] == station_code]
class BootstrapCI():
    def __init__(self) -> None:
        self.lower_bound = None
        self.upper_bound = None
        self.statistic = None
        self.ci = None

    def set_ci(self, ci):
        self.ci = ci

    def get_ci(self):
        return self.ci
    
    def set_n_bootstrap(self, n_bootstrap):
        self.n_bootstrap = n_bootstrap

    def get_n_bootstrap(self):
        return self.n_bootstrap
    
    def set_statistic(self, statistic):
        self.statistic = statistic
    
    def get_statistic(self):
        return self.statistic
    
    def set_lower_bound(self, lower_bound):
        self.lower_bound = lower_bound

    def get_lower_bound(self):
        return self.lower_bound
    
    def set_upper_bound(self, upper_bound):
        self.upper_bound = upper_bound

    def get_upper_bound(self):
        return self.upper_bound
    
    def get_bootstrap_sample(self, data):
        return np.random.choice(data, size=len(data))
    
    def get_statistic_from_sample(self, sample):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def get_bootstrap_replicates(self, data, n_bootstrap):
        bootstrap_replicates = []
        for _ in range(n_bootstrap):
            bootstrap_sample = self.get_bootstrap_sample(data)
            bootstrap_replicates.append(self.get_statistic_from_sample(bootstrap_sample))
        return bootstrap_replicates
    
    def get_ci_from_bootstrap_replicates(self, bootstrap_replicates, alpha):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def get_ci(self, data, n_bootstrap, alpha):
        bootstrap_replicates = self.get_bootstrap_replicates(data, n_bootstrap)
        self.ci = self.get_ci_from_bootstrap_replicates(bootstrap_replicates, alpha)
        return self.ci
    
class BootstrapConfidenceInterval(BootstrapCI):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.n_bootstrap = None
        self.alpha = None
        self.statistic = None
        self.ci = None
    
    def set_statistic(self, statistic):
        self.statistic = statistic

    def set_n_bootstrap(self, n_bootstrap):
        self.n_bootstrap = n_bootstrap

    def get_alpha(self):
        return self.alpha

    def set_alpha(self, alpha):
        self.alpha = alpha

    def get_statistic_from_sample(self, sample):
        if self.statistic == 'mean':
            return np.mean(sample)
        elif self.statistic == 'median':
            return np.median(sample)
        elif self.statistic == 'variance':  
            return np.var(sample)
        else:
            raise ValueError(f"Statistic {self.statistic} is not implemented")
    
    def get_ci_from_bootstrap_replicates(self, bootstrap_replicates, alpha):
        if self.statistic == 'mean':
            return np.percentile(bootstrap_replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        elif self.statistic == 'median':
            return np.percentile(bootstrap_replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        elif self.statistic == 'variance':
            return np.percentile(bootstrap_replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        else:
            raise ValueError(f"Statistic {self.statistic} is not implemented")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Dados de Estações
file_estacoes = "estacoes_data_cluster_0.csv"

# Arquivo de Estações
data_estacoes = DataLoaderEstacoes()
data_estacoes.file = file_estacoes
data_estacoes.load_csv()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# inicializar o estilo do seaborn
sns.set_style("darkgrid")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONFIGURAÇÕES DO PAINEL DE DADOS

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 2. Configurar a página
st.set_page_config(
    page_title="Avaliação de Modelos de ML para Estudos de Regionalização de Vazões",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("##")
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Barra Lateral

# Opções de modelos de ML a serem selecionados na barra lateral
# Dicionário com as opções de modelos de ML e campos de predição de cada um dos modelos
ml_options = {
    'BRR':['resultados_ML_brr_0.csv','y_pred_brr','brr'],
    'KNNR':['resultados_ML_knnr_0.csv','y_pred_knnr','knnr'],
    'Linear Regression':['resultados_ML_lr_0.csv','y_pred_LinearRegression','lr'],
    'Random Forest Regressor':['resultados_ML_rfr_0.csv','y_pred_RandomForestRegressor','rfr'],
    'SVR':['resultados_ML_svr_0.csv', 'y_pred_SVR','svr'],
    'XGBoost Regressor':['resultados_ML_xgbr_0.csv','y_pred_XGBRegressor','xgbr'],
    'XGBoost Randon Forest Regressor':['resultados_ML_xgbrf_0.csv','y_pred_XGBRFRegressor','xgbrf'],
}
# Monta a lista de modelos de ML para seleção na barra lateral
selected_ml = st.sidebar.selectbox("Selecione o modelo de machine learning", list(ml_options.keys()))
# Variável com o nome do arquivo de resultados - resultado da seleção do modelo de ML
file_resultados = ml_options[selected_ml][0]
# Variável com o nome do campo de predição - resultado da seleção do modelo de ML
y_pred = ml_options[selected_ml][1]
# Variável com mnemonico do modelo de ML - resultado da seleção do modelo de ML
mnemonico = ml_options[selected_ml][2]

# Carregar o arquivo de dados de resultados do ML selecionado
data = DataLoaderResultadosML()
data.file = file_resultados
data.load_csv()

# Monta Lista de Códigos de Estações
station_codes = data_estacoes.get_station_codes()
# Criar a seção de seleção de estação para a barra lateral
selected_station = st.sidebar.selectbox("Selecione o código da estação", station_codes)

# 7. Filtrar os dados pela estação selecionada
filtered_data = data.get_filtered_data(selected_station)
filtered_data_estacoes = data_estacoes.get_filtered_data(selected_station)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SEÇÃO DO PAINEL DE DADOS 

# Titulo do painel
with st.container():
    st.title("Painel de Dados")
    st.subheader("Avaliação de Modelos de ML para Estudos de Regionalização de Vazões")

st.markdown("---")

# # container para a tabela de estações
# with st.container():
#     st.write(f"Tabela com os dados da estação - Código da Estação : {selected_station}")
#     st.table(filtered_data_estacoes)

with st.expander(f"⏰ Tabela com os dados da estação - Código da Estação : {selected_station}"):
    showData=st.multiselect('Filter: ',filtered_data_estacoes.columns,default=["codigo","cobacia","cocursodag","q95"])
    st.dataframe(filtered_data_estacoes[showData],use_container_width=True, height=200)

st.markdown("---")

# container para o indicador de vazão, a q95 da estação
with st.container():
    st.write(f"Indicador de Vazão - Q95 da estação - Código da Estação : {selected_station}")
    st.write(filtered_data_estacoes["q95"].values[0])

st.markdown("---")

# kpis com as estatísticas descritivas - 3 colunas
metrics_column1, metrics_column2, metrics_column3 = st.columns(3)
# medidas de centralidade
with metrics_column1:
    st.subheader("Média")
    st.write(filtered_data[y_pred].mean())
    st.subheader("Mediana")
    st.write(filtered_data[y_pred].median())
# medidas de dispersão
with metrics_column2:
    st.subheader("Variância")
    st.write(filtered_data[y_pred].var())
    st.subheader("Desvio Padrão")
    st.write(filtered_data[y_pred].std())
# medidas de intervalode confiança
with metrics_column3:
    st.subheader("lower bound 2,5%")
    st.write(filtered_data[y_pred].quantile(0.025))
    st.subheader("upper bound 97,5%")
    st.write(filtered_data[y_pred].quantile(0.975))

st.markdown("---")

# seção com os testes bootstrap para intervalo de confiança dos quantilles 0,025 e 0,975
st.subheader("Testes bootstrap para intervalo de confiança dos quantilles 0,025 e 0,975")
st.write("Testes bootstrap para intervalo de confiança dos quantilles 0,025 e 0,975")
# criar o objeto bootstrap
bootstrap = BootstrapConfidenceInterval(filtered_data[y_pred])
# setar o número de amostras bootstrap
bootstrap.set_n_bootstrap(1000)
# setar o nível de confiança
bootstrap.set_alpha(0.05)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Sobre a média

# setar a estatística
bootstrap.set_statistic("mean")
# calcular o intervalo de confiança
ci = bootstrap.get_ci(filtered_data[y_pred], bootstrap.get_n_bootstrap(), bootstrap.get_alpha())
# imprimir o intervalo de confiança em 3 colunas
ci_column1, ci_column2, ci_column3 = st.columns(3)
# intervalo de confiança da média na coluna 1
with ci_column1:
    st.subheader("CI da média")
    # setar a estatística
    bootstrap.set_statistic("mean")
    # calcular o intervalo de confiança
    ci = bootstrap.get_ci(filtered_data[y_pred], bootstrap.get_n_bootstrap(), bootstrap.get_alpha())
    st.write("lower bound: ", ci[0])
    st.write("upper bound: ", ci[1])
# intervalo de confiança da mediana na coluna 2
with ci_column2:
    st.subheader("CI da mediana")
    # setar a estatística
    bootstrap.set_statistic("median")
    # calcular o intervalo de confiança
    ci = bootstrap.get_ci(filtered_data[y_pred], bootstrap.get_n_bootstrap(), bootstrap.get_alpha())
    st.write("lower bound: ", ci[0])
    st.write("upper bound: ", ci[1])
# intervalo de confiança da variância na coluna 3
with ci_column3:
    st.subheader("CI da variância")
    # setar a estatística
    bootstrap.set_statistic("variance")
    # calcular o intervalo de confiança
    ci = bootstrap.get_ci(filtered_data[y_pred], bootstrap.get_n_bootstrap(), bootstrap.get_alpha())
    st.write("lower bound: ", ci[0])
    st.write("upper bound: ", ci[1])
st.markdown("---")
# 10. Criar a seção de seleção de gráficos
#graph_types = ["Histograma", "Violinplot", "Boxplot", "Q-Q plot"]
#selected_graph_types = st.sidebar.multiselect("Selecione o(s) tipo(s) de gráfico", graph_types, default=graph_types)

# gráficos em colunas
graph_column1, graph_column2 = st.columns(2)
# sobre o histograma
with graph_column1:
    st.subheader("Histograma")
    st.write("Histograma das previsões do modelo XGBRegressor")
    sns.set(rc={'figure.figsize':(6,6)})
    sns.histplot(filtered_data[y_pred], bins=100)
    #linha vertical para para a q95 da estação
    plt.axvline(x=filtered_data_estacoes["q95"].values[0], color='r', linestyle='--')
    # linha vertical azul para a média encontrada no modelo de ml
    plt.axvline(x=filtered_data[y_pred].mean(), color='b', linestyle='--')
    # linhas azuis para quartilles 0,025 e 0,975
    plt.axvline(x=filtered_data[y_pred].quantile(0.025), color='g', linestyle='--')
    plt.axvline(x=filtered_data[y_pred].quantile(0.975), color='g', linestyle='--')
    st.pyplot()

# sobre o violinplot
with graph_column2:
    st.subheader("Violinplot")
    st.write(f"Violinplot das previsões do modelo {mnemonico}")
    sns.set(rc={'figure.figsize':(6,6)})
    sns.violinplot(y=filtered_data[y_pred])
    # linha horizontal vermelha para para a q95 da estação
    plt.axhline(y=filtered_data_estacoes["q95"].values[0], color='r', linestyle='--')
    # linha horizontal azul para a média encontrada no modelo de ml
    plt.axhline(y=filtered_data[y_pred].mean(), color='b', linestyle='--')
    # linhas azuis para quartilles 0,025 e 0,975
    plt.axhline(y=filtered_data[y_pred].quantile(0.025), color='g', linestyle='--')
    plt.axhline(y=filtered_data[y_pred].quantile(0.975), color='g', linestyle='--')
    st.pyplot()
# graficos em colunas
graph_column3, graph_column4 = st.columns(2)
# sobre o boxplot
with graph_column3:
    st.subheader("Boxplot")
    st.write(f"Boxplot das previsões do modelo {mnemonico}")
    sns.set(rc={'figure.figsize':(6,6)})
    sns.boxplot(y=filtered_data[y_pred])
    st.pyplot()

with graph_column4:
    st.subheader("Q-Q plot")
    st.write(f"Q-Q plot das previsões do modelo {mnemonico}")
    import scipy.stats as stats
    fig, ax = plt.subplots(figsize=(6,6))
    stats.probplot(filtered_data[y_pred], plot=ax)
    st.pyplot(fig)

st.markdown("---")


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# METODO PARA GERAR TODAS AS ESTATÍSTICAS DO PAINEL PARA AS ESTAÇÕES NO data.data 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# DataLoader de arquivo de intervalos de confiança
data_intervalos = DataLoader()
data_intervalos.file = f"intervalos_confianca_{mnemonico}_0.csv"
data_intervalos.load_csv()

st.header(f"Intervalos de Confiança - {mnemonico}")

with st.expander(f"⏰ Estatísticas das Estações para o Modelo de ML selecionado - {mnemonico}"):
    columns_list = data_intervalos.data.columns.tolist()
    showData=st.multiselect('Filter: ',data_intervalos.data.columns,default=columns_list)
    st.dataframe(data_intervalos.data[showData],use_container_width=True, height=200)

# criar botão para fazer download do arquivo de intervalos de confiança
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')
    
csv = convert_df(data_intervalos.data)

st.download_button(
   "Press to Download CSV",
   csv,
   f"estatistica_ML_{mnemonico}.csv",
   "text/csv",
   key='download-csv'
)

st.markdown("---")

st.write("KPIs para os intervalos de confiança das previsões do modelo de ML selecionado")

metrics_column1, metrics_column2, metrics_column3 = st.columns(3)
filtered_names = list(filter(lambda x: "dentro_intervalo" in x, columns_list))

# Quantidades de estações dentro do intervalo de confiança
with metrics_column1:
    st.subheader("Dentro do Intervalo")
    st.write(int(data_intervalos.data[filtered_names].sum()))
# Quantidades de estações fora do intervalo de confiança
with metrics_column2:
    st.subheader("Fora do Intervalo")
    st.write(int(len(data_intervalos.data) - data_intervalos.data[filtered_names].sum()))
# Percentual de estações dentro do intervalo de confiança
with metrics_column3:
    st.subheader("Proporção")
    st.write(float(data_intervalos.data[filtered_names].sum() / len(data_intervalos.data)))
    #st.write(len(filtered_names.data()))

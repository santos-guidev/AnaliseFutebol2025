# Análise de Probabilidades em Jogos de Futebol ⚽

Este projeto calcula probabilidades de placares, estatísticas de desempenho de times e odds justas para apostas esportivas, utilizando o modelo de distribuição de Poisson. Ideal para entusiastas de futebol e traders esportivos.

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

## 🚀 Funcionalidades

- **Cálculo de Probabilidades de Placares** usando distribuição de Poisson.
- **Estatísticas Detalhadas** de times (gols, finalizações, escanteios).
- **Odds Justas** para mercados como Over/Under, Ambas Marcam (BTTS) e Resultado Final.
- **Comparação com Odds Reais** de casas de apostas.
- **Visualização Interativa** de dados com gráficos e tabelas.

## 📦 Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/santos-guidev/AnaliseFutebol2025.git

## Instale as dependências:

pip install -r requirements.txt

## 🛠️ Uso Execute o aplicativo Streamlit:

streamlit run app.py


## Passos no Aplicativo:

Selecione uma Liga (ex: Premier League, La Liga).

Escolha os Times (mandante e visitante).

## Visualize:

Probabilidades de placares.

Estatísticas comparativas.

Odds justas vs. odds reais.

Forma recente dos times.

## 📊 Estrutura do Código - Bibliotecas Principais

streamlit: Interface web.

pandas/numpy: Manipulação de dados.

plotly: Visualizações interativas.

scipy.stats.poisson: Modelagem estatística.

## Funções-Chave
Função	Descrição
load_data(url)	Carrega dados de um arquivo Excel remoto.
get_team_stats(team, df)	Retorna médias de gols, finalizações e escanteios de um time.
calculate_poisson_probabilities()	Calcula a matriz de probabilidades de placares (0-5 gols).
calculate_additional_markets()	Calcula odds para mercados secundários (Over/Under, BTTS).

## Exemplo de Saída
Exemplo de Dashboard <!-- Adicione uma imagem real do seu app -->

## 🌍 Ligas Suportadas
Argentina Primera División

Premier League (Inglaterra)

La Liga (Espanha)

Bundesliga (Alemanha)

Serie A (Brasil/Itália)

...e mais!

## 🤝 Como Contribuir
Faça um fork do projeto.

Crie uma branch (git checkout -b feature/nova-feature).

Commit suas mudanças (git commit -m 'Adiciona nova feature').

Push para a branch (git push origin feature/nova-feature).

Abra um Pull Request.

## 📄 Licença
Distribuído sob a licença MIT. Veja LICENSE para mais detalhes.

Nota: Para dados em tempo real, integre APIs como Transfermarkt ou Football-Data.org.

Quer ajuda para adaptar algo específico? Só pedir! 😊

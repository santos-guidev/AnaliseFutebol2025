import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
import plotly.express as px
from scipy.stats import poisson

#####################################
# Carregamento dos dados
#####################################
def load_data(url):
    response = requests.get(url)
    data = BytesIO(response.content)
    df = pd.read_excel(data)
    
    # Converte a coluna de datas para datetime (se necessário)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

#####################################
# Estatísticas Avançadas
#####################################
def get_team_stats(team, df):
    """
    Retorna estatísticas médias de gols, chutes, escanteios etc.
    para todos os jogos do time (casa e fora).
    """
    team_df_home = df[df['Home'] == team]
    team_df_away = df[df['Away'] == team]
    
    total_matches = len(team_df_home) + len(team_df_away)
    if total_matches == 0:
        return {
            'avg_goals_scored': 0,
            'avg_goals_conceded': 0,
            'avg_shots': 0,
            'avg_shots_on_target': 0,
            'avg_corners': 0
        }

    # Gols marcados
    goals_scored_home = team_df_home['Goals_H_FT'].sum()
    goals_scored_away = team_df_away['Goals_A_FT'].sum()
    avg_goals_scored = (goals_scored_home + goals_scored_away) / total_matches

    # Gols sofridos
    goals_conceded_home = team_df_home['Goals_A_FT'].sum()
    goals_conceded_away = team_df_away['Goals_H_FT'].sum()
    avg_goals_conceded = (goals_conceded_home + goals_conceded_away) / total_matches

    # Finalizações
    shots_home = team_df_home['Shots_H'].sum() if 'Shots_H' in df.columns else 0
    shots_away = team_df_away['Shots_A'].sum() if 'Shots_A' in df.columns else 0
    avg_shots = (shots_home + shots_away) / total_matches

    # Finalizações no alvo
    s_ot_home = team_df_home['ShotsOnTarget_H'].sum() if 'ShotsOnTarget_H' in df.columns else 0
    s_ot_away = team_df_away['ShotsOnTarget_A'].sum() if 'ShotsOnTarget_A' in df.columns else 0
    avg_shots_on_target = (s_ot_home + s_ot_away) / total_matches

    # Escanteios
    corners_home = team_df_home['Corners_H_FT'].sum() if 'Corners_H_FT' in df.columns else 0
    corners_away = team_df_away['Corners_A_FT'].sum() if 'Corners_A_FT' in df.columns else 0
    avg_corners = (corners_home + corners_away) / total_matches

    return {
        'avg_goals_scored': avg_goals_scored,
        'avg_goals_conceded': avg_goals_conceded,
        'avg_shots': avg_shots,
        'avg_shots_on_target': avg_shots_on_target,
        'avg_corners': avg_corners
    }

#####################################
# Forma Recente (Últimos N jogos)
#####################################
def get_recent_form(team, df, n=5):
    """
    Retorna o DataFrame dos últimos n jogos do time,
    além de pontos obtidos e uma lista com resultados (V/E/D).
    """
    team_df = df[(df['Home'] == team) | (df['Away'] == team)].copy()
    if 'Date' in team_df.columns:
        team_df = team_df.sort_values('Date', ascending=False)
    last_n = team_df.head(n)

    points = 0
    results = []
    for _, row in last_n.iterrows():
        home_goals = row['Goals_H_FT']
        away_goals = row['Goals_A_FT']
        if row['Home'] == team:
            if home_goals > away_goals:
                points += 3
                results.append('V')
            elif home_goals == away_goals:
                points += 1
                results.append('E')
            else:
                results.append('D')
        else:
            if away_goals > home_goals:
                points += 3
                results.append('V')
            elif away_goals == home_goals:
                points += 1
                results.append('E')
            else:
                results.append('D')

    return last_n, points, results

#####################################
# Cálculo de Probabilidades (Poisson)
#####################################
def calculate_poisson_probabilities(team1, team2, df):
    """
    Calcula matriz de probabilidades de placares possíveis (0-5 gols),
    com base na média de gols marcados em casa pelo time1
    e na média de gols marcados fora pelo time2.
    """
    home_games = df[df['Home'] == team1]
    away_games = df[df['Away'] == team2]
    
    if home_games.empty or away_games.empty:
        return None
    
    avg_goals_home = home_games['Goals_H_FT'].mean()
    avg_goals_away = away_games['Goals_A_FT'].mean()
    
    max_goals = 5
    probabilities = np.zeros((max_goals + 1, max_goals + 1))
    
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probabilities[home_goals, away_goals] = (
                poisson.pmf(home_goals, avg_goals_home) *
                poisson.pmf(away_goals, avg_goals_away)
            )
    return probabilities

#####################################
# Cálculo de mercados Over/Under e BTTS
#####################################
def calculate_additional_markets(probabilities):
    """
    Retorna dicionário com probabilidades e odds justas de:
    Over/Under 0.5, 1.5, 2.5, 3.5, 4.5 e Ambas Marcam (BTTS).
    """
    results = {}
    lines = [0.5, 1.5, 2.5, 3.5, 4.5]
    
    # Over/Under
    for line in lines:
        prob_over = 0.0
        for i in range(probabilities.shape[0]):
            for j in range(probabilities.shape[1]):
                if (i + j) > line:
                    prob_over += probabilities[i, j]
        prob_under = 1 - prob_over
        
        odds_over = 1/prob_over if prob_over > 0 else float('inf')
        odds_under = 1/prob_under if prob_under > 0 else float('inf')
        
        results[f'Over {line}'] = (prob_over, odds_over)
        results[f'Under {line}'] = (prob_under, odds_under)
    
    # Ambas Marcam (BTTS)
    btts_prob = 0.0
    for i in range(1, probabilities.shape[0]):
        for j in range(1, probabilities.shape[1]):
            btts_prob += probabilities[i,j]
    btts_odds = 1/btts_prob if btts_prob > 0 else float('inf')
    
    nbtts_prob = 1 - btts_prob
    nbtts_odds = 1/nbtts_prob if nbtts_prob > 0 else float('inf')
    
    results['BTTS - Sim'] = (btts_prob, btts_odds)
    results['BTTS - Não'] = (nbtts_prob, nbtts_odds)
    
    return results

#####################################
# Odds Reais x Odds Justas
#####################################
def get_real_odds(team1, team2, df):
    """
    Retorna as odds da partida mais recente para
    Home (Odd_H_FT), Draw (Odd_D_FT) e Away (Odd_A_FT).
    """
    match_df = df[(df['Home'] == team1) & (df['Away'] == team2)]
    if match_df.empty:
        return None
    
    match_df = match_df.sort_values('Date', ascending=False)
    last_match = match_df.iloc[0]
    
    if 'Odd_H_FT' not in last_match or pd.isna(last_match['Odd_H_FT']):
        return None
    
    return {
        'home': last_match['Odd_H_FT'],
        'draw': last_match['Odd_D_FT'],
        'away': last_match['Odd_A_FT']
    }

#####################################
# Ligas disponíveis
#####################################
leagues = {
    "Argentina Primera División": "https://github.com/futpythontrader/YouTube/raw/refs/heads/main/Bases_de_Dados/FootyStats/Bases_de_Dados_(2022-2025)/Argentina%20Primera%20Divisi%C3%B3n_2025.xlsx",
    "Belgium Pro League": "https://github.com/futpythontrader/YouTube/raw/refs/heads/main/Bases_de_Dados/FootyStats/Bases_de_Dados_(2022-2025)/Belgium%20Pro%20League_20242025.xlsx",
    "England Championship": "https://github.com/futpythontrader/YouTube/raw/refs/heads/main/Bases_de_Dados/FootyStats/Bases_de_Dados_(2022-2025)/England%20Championship_20242025.xlsx",
    "England EFL League One": "https://github.com/futpythontrader/YouTube/raw/refs/heads/main/Bases_de_Dados/FootyStats/Bases_de_Dados_(2022-2025)/England%20EFL%20League%20One_20242025.xlsx",
    "England Premier League": "https://github.com/futpythontrader/YouTube/raw/refs/heads/main/Bases_de_Dados/FootyStats/Bases_de_Dados_(2022-2025)/England%20Premier%20League_20242025.xlsx",
    "France Ligue 1": "https://github.com/futpythontrader/YouTube/raw/refs/heads/main/Bases_de_Dados/FootyStats/Bases_de_Dados_(2022-2025)/France%20Ligue%201_20242025.xlsx",
    "France Ligue 2": "https://github.com/futpythontrader/YouTube/raw/refs/heads/main/Bases_de_Dados/FootyStats/Bases_de_Dados_(2022-2025)/France%20Ligue%202_20242025.xlsx",
    "Germany 2. Bundesliga": "https://github.com/futpythontrader/YouTube/raw/refs/heads/main/Bases_de_Dados/FootyStats/Bases_de_Dados_(2022-2025)/Germany%202.%20Bundesliga_20242025.xlsx",
    "Germany Bundesliga": "https://github.com/futpythontrader/YouTube/raw/refs/heads/main/Bases_de_Dados/FootyStats/Bases_de_Dados_(2022-2025)/Germany%20Bundesliga_20242025.xlsx",
    "Italy Serie A": "https://github.com/futpythontrader/YouTube/raw/refs/heads/main/Bases_de_Dados/FootyStats/Bases_de_Dados_(2022-2025)/Italy%20Serie%20A_20232024.xlsx",
    "Japan J1 League": "https://github.com/futpythontrader/YouTube/raw/refs/heads/main/Bases_de_Dados/FootyStats/Bases_de_Dados_(2022-2025)/Japan%20J1%20League_2024.xlsx",
    "Netherlands Eerste Divisie": "https://github.com/futpythontrader/YouTube/raw/refs/heads/main/Bases_de_Dados/FootyStats/Bases_de_Dados_(2022-2025)/Netherlands%20Eerste%20Divisie_20242025.xlsx",
    "Portugal Liga NOS": "https://github.com/futpythontrader/YouTube/raw/refs/heads/main/Bases_de_Dados/FootyStats/Bases_de_Dados_(2022-2025)/Portugal%20Liga%20NOS_20242025.xlsx",
    "Portugal LigaPro": "https://github.com/futpythontrader/YouTube/raw/refs/heads/main/Bases_de_Dados/FootyStats/Bases_de_Dados_(2022-2025)/Portugal%20LigaPro_20242025.xlsx",
    "Spain La Liga": "https://github.com/futpythontrader/YouTube/raw/refs/heads/main/Bases_de_Dados/FootyStats/Bases_de_Dados_(2022-2025)/Spain%20La%20Liga_20242025.xlsx",
    "Turkey Süper Lig": "https://github.com/futpythontrader/YouTube/raw/refs/heads/main/Bases_de_Dados/FootyStats/Bases_de_Dados_(2022-2025)/Turkey%20S%C3%BCper%20Lig_20242025.xlsx",
    "USA MLS": "https://github.com/futpythontrader/YouTube/raw/refs/heads/main/Bases_de_Dados/FootyStats/Bases_de_Dados_(2022-2025)/USA%20MLS_2025.xlsx"
}

#####################################
# Interface Streamlit
#####################################
st.title("Dashboard de Probabilidades e Odds Justas - Versão Avançada (Sem Jogos do Dia)")

# Seleciona o campeonato
selected_league = st.selectbox("Selecione o Campeonato", list(leagues.keys()))
df = load_data(leagues[selected_league])

# Seleciona times
teams = sorted(set(df['Home'].unique()) | set(df['Away'].unique()))
team1 = st.selectbox("Time da Casa", teams)
team2 = st.selectbox("Time Visitante", teams)

if team1 and team2 and team1 != team2:
    # =======================
    # 1) Forma Recente
    # =======================
    st.subheader("Forma Recente (Últimos 5 Jogos)")

    recent_team1, points_t1, results_t1 = get_recent_form(team1, df, n=5)
    recent_team2, points_t2, results_t2 = get_recent_form(team2, df, n=5)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{team1}** - Últimos 5 Jogos: {results_t1} (Total de {points_t1} pts)")
        st.dataframe(recent_team1[['Date','Home','Away','Goals_H_FT','Goals_A_FT']])
    with col2:
        st.markdown(f"**{team2}** - Últimos 5 Jogos: {results_t2} (Total de {points_t2} pts)")
        st.dataframe(recent_team2[['Date','Home','Away','Goals_H_FT','Goals_A_FT']])

    # =======================
    # 2) Estatísticas Avançadas
    # =======================
    st.subheader("Estatísticas Avançadas (Médias por Jogo)")
    stats_team1 = get_team_stats(team1, df)
    stats_team2 = get_team_stats(team2, df)

    comparison_df = pd.DataFrame({
        'Métrica': [
            'Gols Marcados',
            'Gols Sofridos',
            'Finalizações',
            'Chutes no Alvo',
            'Escanteios'
        ],
        team1: [
            stats_team1['avg_goals_scored'],
            stats_team1['avg_goals_conceded'],
            stats_team1['avg_shots'],
            stats_team1['avg_shots_on_target'],
            stats_team1['avg_corners']
        ],
        team2: [
            stats_team2['avg_goals_scored'],
            stats_team2['avg_goals_conceded'],
            stats_team2['avg_shots'],
            stats_team2['avg_shots_on_target'],
            stats_team2['avg_corners']
        ]
    })

    st.dataframe(comparison_df.style.format(precision=2))

    fig_comp = px.bar(
        comparison_df,
        x='Métrica',
        y=[team1, team2],
        barmode='group',
        labels={'value': 'Média', 'Métrica': 'Estatísticas'},
        title='Comparação de Desempenho'
    )
    st.plotly_chart(fig_comp)

    # =======================
    # 3) Probabilidades (Poisson)
    # =======================
    st.subheader("Probabilidades de Placar (Distribuição Poisson)")
    probabilities = calculate_poisson_probabilities(team1, team2, df)
    
    if probabilities is not None:
        df_prob = pd.DataFrame(
            probabilities,
            columns=[f"{i} Gols Visitante" for i in range(probabilities.shape[1])],
            index=[f"{i} Gols Casa" for i in range(probabilities.shape[0])]
        )
        
        st.write("**Matriz de Probabilidades**")
        st.dataframe(df_prob.style.format(precision=3))
        
        fig_heatmap = px.imshow(
            probabilities,
            text_auto=".2f",
            labels=dict(color="Probabilidade"),
            x=[f"{i} Gols Visitante" for i in range(probabilities.shape[1])],
            y=[f"{i} Gols Casa" for i in range(probabilities.shape[0])],
            color_continuous_scale="Blues",
            title="Heatmap de Probabilidades de Placar"
        )
        st.plotly_chart(fig_heatmap)

        # Probabilidade de vitória/empate/derrota
        home_win_prob = np.sum(np.tril(probabilities, -1))
        draw_prob = np.sum(np.diag(probabilities))
        away_win_prob = np.sum(np.triu(probabilities, 1))

        odds_home = 1 / home_win_prob if home_win_prob > 0 else float("inf")
        odds_draw = 1 / draw_prob if draw_prob > 0 else float("inf")
        odds_away = 1 / away_win_prob if away_win_prob > 0 else float("inf")

        st.write("**Probabilidades de Resultado Final**")
        col1_res, col2_res, col3_res = st.columns(3)
        col1_res.metric("Vitória Casa", f"{home_win_prob:.2%}", f"Odds Justa: {odds_home:.2f}")
        col2_res.metric("Empate", f"{draw_prob:.2%}", f"Odds Justa: {odds_draw:.2f}")
        col3_res.metric("Vitória Visitante", f"{away_win_prob:.2%}", f"Odds Justa: {odds_away:.2f}")

        # =======================
        # 3.1) Mercados Alternativos: Over/Under e BTTS
        # =======================
        st.subheader("Mercados Alternativos (Over/Under e Ambas Marcam)")
        additional = calculate_additional_markets(probabilities)
        
        lines = [0.5, 1.5, 2.5, 3.5, 4.5]
        for line in lines:
            over_key = f"Over {line}"
            under_key = f"Under {line}"
            over_prob, over_odds = additional[over_key]
            under_prob, under_odds = additional[under_key]
            
            c1, c2 = st.columns(2)
            c1.metric(over_key, f"{over_prob:.2%}", f"Odds: {over_odds:.2f}")
            c2.metric(under_key, f"{under_prob:.2%}", f"Odds: {under_odds:.2f}")
            st.write("---")

        btts_yes_prob, btts_yes_odds = additional['BTTS - Sim']
        btts_no_prob, btts_no_odds = additional['BTTS - Não']
        col_btts1, col_btts2 = st.columns(2)
        col_btts1.metric("Ambas Marcam (Sim)", f"{btts_yes_prob:.2%}", f"Odds: {btts_yes_odds:.2f}")
        col_btts2.metric("Ambas Marcam (Não)", f"{btts_no_prob:.2%}", f"Odds: {btts_no_odds:.2f}")

        # =======================
        # 3.2) Tabela Consolidada de Odds Justas e Probabilidades
        # =======================
        st.subheader("Tabela Consolidada de Probabilidades e Odds Justas")
        df_markets = []

        # 1X2
        df_markets.append({"Mercado": "Casa",       "Prob (%)": home_win_prob*100, "Odds Justa": odds_home})
        df_markets.append({"Mercado": "Empate",     "Prob (%)": draw_prob*100,     "Odds Justa": odds_draw})
        df_markets.append({"Mercado": "Visitante",  "Prob (%)": away_win_prob*100, "Odds Justa": odds_away})

        # Over/Under
        for line in lines:
            over_key = f"Over {line}"
            under_key = f"Under {line}"
            over_prob, over_odds = additional[over_key]
            under_prob, under_odds = additional[under_key]
            df_markets.append({"Mercado": over_key,   "Prob (%)": over_prob*100,   "Odds Justa": over_odds})
            df_markets.append({"Mercado": under_key,  "Prob (%)": under_prob*100,  "Odds Justa": under_odds})

        # BTTS
        df_markets.append({"Mercado": "BTTS - Sim", "Prob (%)": btts_yes_prob*100, "Odds Justa": btts_yes_odds})
        df_markets.append({"Mercado": "BTTS - Não", "Prob (%)": btts_no_prob*100,  "Odds Justa": btts_no_odds})

        df_markets = pd.DataFrame(df_markets)
        st.dataframe(df_markets.style.format({"Prob (%)": "{:.2f}", "Odds Justa": "{:.2f}"}))

        # =======================
        # 4) Odds Reais x Odds Justas (Último Confronto)
        # =======================
        st.subheader("Odds Reais x Odds Justas (última partida do confronto)")
        real_odds = get_real_odds(team1, team2, df)
        if real_odds:
            st.write(f"**Odds da base (Último {team1} x {team2} encontrado):**")
            col1_odds, col2_odds, col3_odds = st.columns(3)
            col1_odds.metric("Odd Real Casa", f"{real_odds['home']:.2f}", 
                             f"Dif: {(real_odds['home'] - odds_home):.2f}")
            col2_odds.metric("Odd Real Empate", f"{real_odds['draw']:.2f}",
                             f"Dif: {(real_odds['draw'] - odds_draw):.2f}")
            col3_odds.metric("Odd Real Visitante", f"{real_odds['away']:.2f}",
                             f"Dif: {(real_odds['away'] - odds_away):.2f}")
        else:
            st.info("Não foram encontradas odds reais para este confronto na base.")

    else:
        st.error("Não foi possível calcular probabilidades (faltam dados para este confronto).")

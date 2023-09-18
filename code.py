# Assumptions:
# Defensive possessions are equal to opponent's offensive rebound chances.
# Defensive rebounds are estimated as opponent's offensive rebound chances minus opponent's offensive rebounds.
# Defensive rating is calculated as the ratio of opponent's offensive rebounds to opponent's offensive rebound chances.
# predicating winning and losing chances based on historical data and assumpting values using mean and average

# ========================= Tranforming team 2022 and team stats datasets===================================
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
team_stats = pd.read_csv('team_stats.csv')
df1 = pd.read_csv('team_rebounding_data_22.csv')

# Convert 'gamedate' to datetime and extract 'season'
df1['gamedate'] = pd.to_datetime(df1['gamedate'])
df1['season'] = df1['gamedate'].dt.year

# Calculate team_counts
team_counts = df1.groupby(['team','season']).agg({'offensive_rebounds':'sum', 'off_rebound_chances':'sum','game_number':'count'}).reset_index()
team_counts.rename(columns={'game_number':'games'}, inplace=True)
team_counts['off_rtg'] = ((team_counts['offensive_rebounds'] / team_counts['off_rebound_chances']) * 100).fillna(0)
team_counts['defensive_possessions'] = team_counts['team'].map(df1.groupby('opp_team')['off_rebound_chances'].sum())
team_counts['defensive_rebounds'] = team_counts['team'].map(df1.groupby('opp_team')['off_rebound_chances'].sum()) - team_counts['team'].map(df1.groupby('opp_team')['offensive_rebounds'].sum())
team_counts['def_rtg'] = ((team_counts['defensive_rebounds'] / team_counts['defensive_possessions']) * 100).fillna(0)
team_counts['net_rtg'] = team_counts['off_rtg'] - team_counts['def_rtg']
team_counts = team_counts.drop(['offensive_rebounds', 'off_rebound_chances', 'defensive_possessions', 'defensive_rebounds'], axis=1)

#applying nbateamid based on team
if 'team' not in team_stats.columns:
    team_stats['team'] = ''

new_team_stats = team_counts.copy()
new_team_stats['nbateamid'] = ''

for index, row in new_team_stats.iterrows():
    team_name = row['team']
    if not team_name:
        continue
    matching_team = team_stats[team_stats['team'] == team_name]
    if not matching_team.empty:
        new_team_stats.at[index, 'nbateamid'] = matching_team.iloc[0]['nbateamid']

new_team_stats_2022 = new_team_stats[new_team_stats['season'] == 2022]
new_team_stats_2023 = new_team_stats[new_team_stats['season'] == 2023]
updated_team_stats = pd.concat([team_stats,new_team_stats_2022, new_team_stats_2023], ignore_index=True)

#predicting wins and losses
avg_wins_all = updated_team_stats.groupby('team')['W'].mean()
avg_losses_all = updated_team_stats.groupby('team')['L'].mean()

season_2022_2023 = updated_team_stats[(updated_team_stats['season'] == 2022) | (updated_team_stats['season'] == 2023)]
for index, row in season_2022_2023.iterrows():
    team_name = row['team']
    if pd.isna(row['W']) and team_name in avg_wins_all:
        season_2022_2023.at[index, 'W'] = avg_wins_all[team_name]
    if pd.isna(row['L']) and team_name in avg_losses_all:
        season_2022_2023.at[index, 'L'] = avg_losses_all[team_name]

updated_team_stats.update(round(season_2022_2023))
# updated_team_stats.to_csv('team_stats_merged_data.csv', index=False)
# print(updated_team_stats)
# ======================================= awards data ======================================================
df = pd.read_csv('awards_data.csv')
seasons_played = df.groupby("nbapersonid")['season'].count().reset_index(name="Seasons Played")
df = df.merge(seasons_played, on="nbapersonid")

df["All NBA Team Percentage"] = ((df["All NBA First Team"] + df["All NBA Second Team"] + df["All NBA Third Team"]) / df["Seasons Played"]) * 100
df["All Defensive Team Percentage"] = ((df["All NBA Defensive First Team"] + df["All NBA Defensive Second Team"]) / df["Seasons Played"]) * 100

df["All Rookie Team Selection"] = "None"
df.loc[df["All Rookie First Team"] == 1, "All Rookie Team Selection"] = "First Team"
df.loc[df["All Rookie Second Team"] == 1, "All Rookie Team Selection"] = "Second Team"

def player_Of(group):
    result = []
    for _, row in group.iterrows():
        Player_Of_The_Month = row['Player Of The Month']
        Player_Of_The_Week = row['Player Of The Week']
        Rookie_Of_The_Month = row['Rookie Of The Month']
        
        if pd.notna(Player_Of_The_Month) and Player_Of_The_Month > 0:
            result.append(f"{int(Player_Of_The_Month)} {'time' if Player_Of_The_Month ==1 else 'times'} Player of the Month")
        if pd.notna(Player_Of_The_Week) and Player_Of_The_Week > 0:
            result.append(f"{int(Player_Of_The_Week)} {'time' if Player_Of_The_Week ==1 else 'times'} Player of the Week")
        if pd.notna(Rookie_Of_The_Month) and Rookie_Of_The_Month > 0:
            result.append(f"{int(Rookie_Of_The_Month)} {'time' if Rookie_Of_The_Month ==1 else 'times'} Rookie of the Month")
    if not result:
        result.append('None')
    return ' and '.join(result)
res1_df= df.groupby(['nbapersonid','season']).apply(player_Of).reset_index(name='Player Of The')
#=========================team selection function===================================================
def team_select(row):
    if row['all_star_game'] and row['rookie_all_star_game']:
        return 'Selected for both'
    elif row['all_star_game']:
        return 'Selected for allstar'
    elif row['rookie_all_star_game']:
        return 'Selected for rookieallstar'
    else:
        return 'None'
df['Team Selected'] = df.apply(team_select, axis=1)
#============================ranking function====================================================
def ranking(row):
    if pd.notna(row['allstar_rk']):
        rank = int(row['allstar_rk'])
        if rank <= 9:
            return 'High Rank'
        elif rank >9 and rank<=4:
            return 'Mid Rank'
        elif rank >3:
            return 'Low Rank'
    return 'No Rank'
df['Ranking'] = df.apply(ranking, axis=1)

# df = df.drop(["All NBA First Team","All NBA Second Team","All NBA Third Team","All NBA Defensive First Team","All NBA Defensive Second Team","All Rookie First Team","All Rookie Second Team","Player Of The Month","Player Of The Week","Rookie Of The Month","all_star_game","rookie_all_star_game","allstar_rk"], axis=1)
df = df.merge(res1_df, on=['nbapersonid','season'])
column_order = [0,1,10,11,12,13,14,15,16,2,3,4,5,6,7,8,9]
df = df.iloc[:, column_order]
df = df.fillna(0)
df = df.round(2)
# df.to_csv('awards_cleaned_data.csv', index=False)
# print(df.columns)
#================================= merging team_stats and awards==================================
# team_stats_awards = pd.merge(updated_team_stats,df,on="season", how='outer')
# team_stats_awards.drop_duplicates()
# # team_stats_awards.to_csv('team_stats and awards.csv', index=False)
# # print(team_stats_awards)
#================================= players cleaning===============================================
players_df = pd.read_csv('player_stats.csv')
players_df = players_df.fillna(0)
players_df['fg_missed'] = players_df['fga'] - players_df['fgm']
players_df['fgm3_missed'] = players_df['fga3'] - players_df['fgm3']
players_df['fgm2_missed'] = players_df['fga2'] - players_df['fgm2']
players_df['3p%'] = (players_df['fgm3']/ players_df['fga3'])*100
players_df['2p%'] = (players_df['fgm2']/ players_df['fga2'])*100
players_df['Total Field Goals'] = players_df['fgm2'] + players_df['fgm3']
players_df['Total Field Goals Attempted'] = players_df['fga2'] + players_df['fga3']
players_df['Scoring Efficiency'] = (players_df['ftm'] + players_df['fgm'] - players_df['tov']) / players_df['fga']
players_df['Rebound Efficiency'] = players_df['tot_reb']/(players_df['off_reb'] + players_df['def_reb']) *100
players_df['Assist-to-Turnover Ratio'] = players_df['ast'] / players_df['tov']
players_df['Steals+Blocks'] = players_df['steals'] + players_df['blocks']
players_df['Total Contributions'] = players_df['ast'] + players_df['steals'] + players_df['blocks']
players_df = players_df.round(2)
column_order = [0,1,2,3,4,5,6,7,8,9,10,11,49,12,13,14,50,15,16,17,51,18,52,53,54,55,19,20,21,22,33,56,23,24,25,34,35,36,57,26,37,27,38,28,39,29,40,58,59,30,31,41,60,42,43,44,45,46,47,48]
players_df = players_df.iloc[:, column_order]
players_df = players_df.fillna(0)
# players_df.to_csv('player_stats_cleaned.csv', index=False)
# print(players_df)
#==================================== visulaization========================================================
#==================================== Avg offensive and defensive rating by team===========================
# avg_off_rating = updated_team_stats['off_rtg'].mean()
# avg_def_rating = updated_team_stats['def_rtg'].mean()
# nba_teams = updated_team_stats['team'].unique()
# bar_width = 0.35
# opacity = 0.8
# index = range(len(nba_teams))

# bar1 = plt.bar(index, [avg_off_rating] * len(nba_teams), bar_width, alpha=opacity, color='b', label='Average Offensive Rating')
# bar2 = plt.bar([i + bar_width for i in index], [avg_def_rating] * len(nba_teams), bar_width, alpha=opacity, color='r', label='Average Defensive Rating')

# plt.xlabel('NBA Teams')
# plt.ylabel('Average Rating')
# plt.title('Average Offensive and Defensive Ratings for NBA Teams Across Seasons')
# plt.legend()
# # plt.show()
# #===================================== Best and worst defensive teams =================================
# best_def_team = updated_team_stats.nlargest(5, 'def_rtg')
# worst_def_team = updated_team_stats.nsmallest(5, 'def_rtg')

# def_ratings = best_def_team['def_rtg'].tolist() + worst_def_team['def_rtg'].to_list()
# team_labels = best_def_team['team'].tolist() + worst_def_team['team'].tolist()

# fig, ax = plt.subplots()
# ax.pie(def_ratings, labels=team_labels, autopct='%1.1f%%')

# plt.title('Best and Worst Defensive Teams')
# plt.axis('equal') 
# # plt.show()
# #=====================================Distribution of NBA awards=======================================
# award_counts_by_season = df.groupby(['season', 'Most Valuable Player_rk', 'Rookie Of The Year_rk', 'Defensive Player Of The Year_rk']).size().reset_index(name='Award Count')
# award_counts_pivot = award_counts_by_season.pivot(index='season', columns=['Most Valuable Player_rk', 'Rookie Of The Year_rk', 'Defensive Player Of The Year_rk'], values='Award Count').fillna(0)
# award_counts_pivot.plot(kind='bar', stacked=True, figsize=(12, 6))
# plt.xlabel('Season')
# plt.ylabel('Count of Awards')
# plt.title('Distribution of NBA Awards by Season')
# plt.legend(title='Award Type', labels=['MVP', 'Rookie of the Year', 'Defensive Player of the Year'])
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()
# #====================================Relation between All NBA team and All-defensive Team percentage==========================
# plt.figure(figsize=(10, 6))
# plt.scatter(df['All NBA Team Percentage'], df['All Defensive Team Percentage'], alpha=0.5)
# plt.xlabel('All-NBA Team Percentage')
# plt.ylabel('All-Defensive Team Percentage')
# plt.title('Relationship between All-NBA Team Percentage and All-Defensive Team Percentage')
# plt.grid()
# plt.show()
# #=====================================Players scoring efficiency===========================================================
# plt.figure(figsize=(10, 6))
# plt.scatter(players_df['efg'], players_df['points'], alpha=0.5)
# plt.xlabel('Effective Field Goal Percentage (eFG)')
# plt.ylabel('Total Points Scored')
# plt.title('Relationship between eFG and Total Points Scored')
# plt.grid()
# plt.show()
# #====================relation between players usage and their contributions like offensive and defensive====================
# plt.figure(figsize=(10, 6))
# plt.scatter(players_df['usg'], players_df['Total Contributions'], alpha=0.5, label='Total Contributions')
# plt.scatter(players_df['usg'], players_df['OWS'] + players_df['DWS'], alpha=0.5, label='Offensive + Defensive Contributions', marker='x', color='red')
# plt.xlabel('Usage Rate (usg)')
# plt.ylabel('Contributions')
# plt.title('Relationship between Usage Rate and Total Contributions (Offense and Defense)')
# plt.grid()
# plt.legend()
# plt.show()

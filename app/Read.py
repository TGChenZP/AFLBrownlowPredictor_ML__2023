import pandas as pd
import json

def read_leaderboard():
    df = pd.read_csv('../presentables/leaderboard.csv')
    df = df.head(30)
    return df



def read_game_by_game_prediction():
    with open('../presentables/game_by_game_prediction.json') as f:
        data = json.load(f)
    
    VOTES = ['3', '2', '1']

    new_out = dict()
    for round in data:
        new_out[round] = dict()
        for game in data[round]:
        
            players = [data[round][game]['3'], data[round][game]['2'], data[round][game]['1']]
            tmp_df = pd.DataFrame({'Votes': VOTES, 'Player': players})
            new_out[round][game] = tmp_df
    
    return new_out
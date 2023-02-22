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
    
    new_out = sort_by_round(new_out)

    return new_out


def sort_by_round(new_out):

    tmp = [(int(key.split()[1]),new_out[key]) for key in new_out]
    tmp.sort(key = lambda x:x[0])
    out = {f'Round {x[0]}':x[1] for x in tmp}

    return out

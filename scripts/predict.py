import pandas as pd
import pickle
import json
import os
from collections import defaultdict as dd
import sys



with open(f'../models/final_models/model1.pickle', 'rb') as f:
    model1 = pickle.load(f)

with open(f'../models/final_models/model2.pickle', 'rb') as f:
    model2 = pickle.load(f)

with open(f'../models/final_models/model3.pickle', 'rb') as f:
    model3 = pickle.load(f)


with open(f'../models/AFL_pipeline_N.pickle', 'rb') as f:
    sanmin = pickle.load(f)
    
model3_COLS = sanmin.final_features['3']
model2_COLS = sanmin.final_features['2']
model1_COLS = sanmin.final_features['1']

model3_COLS = [x for x in model3_COLS if x not in ['3', '2', '1']]
model2_COLS = [x for x in model2_COLS if x not in ['3', '2', '1']]
model1_COLS = [x for x in model1_COLS if x not in ['3', '2', '1']]



manip_type = 'NormalisedData'

csv_list = os.listdir(f'../future data/curated/{manip_type}')
csv_list.sort()

def predict_brownlow(csv_list):
    json_dict = dict()

    tally = dd(int)

    data = pd.DataFrame()
    for file in csv_list:
        if file[-4:] != '.csv':
            continue

        game_dict = dict()
        if str(sys.argv[1]) in file:
            
            round = file.split()[2]
            team1 = file.split()[3]
            team2 = file.split()[5]
            game = team1 + ' v ' + team2

            data = pd.read_csv(f'../future data/curated/{manip_type}/{file}')
            print(file)
            player = data['Player']
            pred3 = model3.predict(data[model3_COLS])
            pred2 = model2.predict(data[model2_COLS])
            pred1 = model1.predict(data[model1_COLS])
            pred = pd.DataFrame({'player': player, '3': pred3, '2': pred2, '1': pred1})

            three_votes = list(pred.sort_values('3', ascending = False)['player'])[0]

            two_votes = list(pred.sort_values('2', ascending = False)['player'])[0]
            if two_votes == three_votes:
                two_votes = list(pred.sort_values('2', ascending = False)['player'])[1]

            one_vote = list(pred.sort_values('1', ascending = False)['player'])[0]
            if one_vote in (three_votes, two_votes):
                one_vote = list(pred.sort_values('2', ascending = False)['player'])[1]
                
                if one_vote in (three_votes, two_votes):
                    one_vote = list(pred.sort_values('2', ascending = False)['player'])[2]
            
            game_dict[3] = three_votes
            game_dict[2] = two_votes
            game_dict[1] = one_vote

            if f'Round {round}' in json_dict:
                json_dict[f'Round {round}'][game] = game_dict
            else:
                json_dict[f'Round {round}'] = dict()
                json_dict[f'Round {round}'][game] = game_dict

            tally[three_votes] += 3
            tally[two_votes] += 2
            tally[one_vote] += 1
    
    return json_dict, tally


json_dict, tally = predict_brownlow(csv_list) 


with open('../presentables/game_by_game_prediction.json', 'w') as f:
    json.dump(json_dict, f, indent=2)


tally_list = list(tally.items())
tally_list.sort(key = lambda x:x[1], reverse=True)
tally_df = pd.DataFrame(tally_list, columns = ['Player', 'Votes'], index = [i+1 for i in range(len(tally_list))])
tally_df['Ranking'] = tally_df.index
tally_df = tally_df[['Ranking', 'Player', 'Votes']]
tally_df.to_csv('../presentables/leaderboard.csv', index = False)
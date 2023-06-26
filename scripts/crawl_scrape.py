import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
import random
from urllib.parse import urljoin
import os
import sys


def scrape(url):
    """ Function which scrapes the given FootyWire webpage """
    
    # Scrape first page
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    
    section = soup.find(id='contentpagecell')
    results = section.findNext('table')
    rows = results.find_all('tr')
    
    
    
    # Check that this is a Home and Away game rather than finals. If not then just stop the process
    findround = section.findNext('table').findNext('table').findNext('table')
    if not bool(re.search(r'Round', str(findround))):
        return False, False, False    # because our normal return returns three objects
    
    
    
    # # Get Brownlow votes data (but input is later)
    # brownlow = list() #Mechanism to prevent non-assignment because some pages don't have bronwlow data
    # for i in range(len(rows)):
    #     if re.search(r'Brownlow Votes:', str(rows[i])):
    #         brownlow = rows[i]
    # if brownlow:
    #     players = brownlow.find_all('a')
    #     brownlow = list()
    #     for player in players:
    #         brownlow.append(re.findall(r'>. .+<', str(player))[0].strip('<>'))
    
    
    
    # Get raw player statistics
    results = soup.find(id='matchscoretable').findNext('table')
    rows=results.find_all('tr')
   
    tick = 0
    for i in range(1, len(rows)): # recent update of website: 0th item is a table of both teams - messes up our structure
        if not tick and len(rows[i].find_all('tr'))>24: # From experiment, blocks that contain player data (what we want) has more than 25 rows
            team1stats=rows[i]
            tick=1
        elif len(rows[i].find_all('tr'))>24:
            team2stats=rows[i]
            break
            
    team1playerstats = team1stats.find_all('tr')[2].find_all('tr') # From experimenting with scraped blob
    team2playerstats = team2stats.find_all('tr')[2].find_all('tr')

    gamestatscol1 = getdata(team1playerstats)
    gamestatscol2 = getdata(team2playerstats)
    
    
    
    # Get advanced player statistics
    time.sleep(random.uniform(0.5, 1))    # First sleep for a random amount of time - trying to hide crawler activity
    
    urlAdv = url + '&advv=Y' # Because advanced statistic's URL is only different from orig URL by this string
    pageAdv = requests.get(urlAdv)
    soupAdv = BeautifulSoup(pageAdv.text, 'html.parser')
    
    resultsAdv = soupAdv.find(id='matchscoretable').findNext('table')
    rowsAdv=resultsAdv.find_all('tr')

    tick = 0
    for i in range(1, len(rowsAdv)): # recent update of website: 0th item is a table of both teams - messes up our structure
        if not tick and len(rowsAdv[i].find_all('tr'))>24:
            team1stats=rowsAdv[i]
            tick=1
        elif len(rowsAdv[i].find_all('tr'))>24:
            team2stats=rowsAdv[i]
            break  
    
    team1playerstats = team1stats.find_all('tr')[2].find_all('tr')
    team2playerstats = team2stats.find_all('tr')[2].find_all('tr')
    
    gamestatscol1A = getadvdata(team1playerstats)
    gamestatscol2A = getadvdata(team2playerstats)
    
    
    
    # Append the advanced player statistics to standard player statistics
    origattb1 = list()
    for i in range(len(gamestatscol1)):
        origattb1.append(gamestatscol1[i][0])

    for i in range(len(gamestatscol1A)):
        if gamestatscol1A[i][0] not in origattb1:
            gamestatscol1.append(gamestatscol1A[i])
            
            
    origattb2 = list()
    for i in range(len(gamestatscol2)):
        origattb2.append(gamestatscol2[i][0])

    for i in range(len(gamestatscol2A)):
        if gamestatscol2A[i][0] not in origattb2:
            gamestatscol2.append(gamestatscol2A[i])
    
    
    
    # Find Winloss data and also records whether a player is from the home team or away team (helps manipulate stats by own team later)
    winloss = section.findNext('table').findNext('table').findNext('table') 
    keyword = re.findall(r'>\n.*\n<', str(winloss.find_all('td')[0]))[0].strip('>\n\n<')

    v1 = int()
    v2 = int()

    if 'defeats' in keyword:
        v1 = 1
        v2 = 0
    elif 'defeated by' in keyword:
        v1 = 0
        v2 = 1
    else:
        v1 = 0.5
        v2 = 0.5
    
    winloss1 = ['Winloss']
    winloss2 = ['Winloss']
    homeaway1 = ['HomeAway']
    homeaway2 = ['HomeAway']
    for i in range(1, len(gamestatscol1[0])):
        winloss1.append(v1)
        homeaway1.append('Home')
    for i in range(1, len(gamestatscol2[0])):
        winloss2.append(v2)
        homeaway2.append('Away')
    
    gamestatscol1.append(winloss1)
    gamestatscol2.append(winloss2)
    
    gamestatscol1.append(homeaway1)
    gamestatscol2.append(homeaway2)
    

    # Collect some statistics for naming the file such as year, round, and team names (collectively metadata)
    Year = re.findall(r'\d{4},', str(winloss))[0].strip(',')
    Round = re.findall(r'Round \d+', str(winloss))[0]
    

    team1 = keyword.split()[0]
    # Hardcoded some of the two worded team names
    if team1 == 'Gold':
        team1 = 'GoldCoast'
    
    elif team1 == 'North':
        team1 = 'NorthMelbourne'
    
    elif team1 == 'Port':
        team1 = 'PortAdelaide'
    
    elif team1 == 'St':
        team1 = 'StKilda'
    
    elif team1 == 'West':
        team1 = 'WestCoast'
    
    elif team1 == 'Western':
        team1 = 'WesternBulldogs'
    
    
    team2 = keyword.split()[-1]
    if team2 == 'Coast': 
        if keyword.split()[-2] == 'Gold':
            team2 = 'GoldCoast'
        elif keyword.split()[-2] == 'West':
            team2 = 'WestCoast'
    
    elif team2 == 'Melbourne' and keyword.split()[-2] == 'North':
        team2 = 'NorthMelbourne'
    
    elif team2 == 'Adelaide' and keyword.split()[-2] == 'Port':
        team2 = 'PortAdelaide'
    
    elif team2 == 'Kilda':
        team2 = 'StKilda'
        
    elif team2 == 'Bulldogs':
        team2 = 'WesternBulldogs'

    if f'../future data/NormalisedData/{Year} Round {Round} {team1} v {team2} (N).csv' in PROCESSED_FILELIST:
        return False, False, False
    
    # Add Brownlow Data into our dataframe as a column
    team1playerlist = gamestatscol1[0]
    team2playerlist = gamestatscol2[0]
    
    # if brownlow:
    #     brownlow[0] = shorten_surname(brownlow[0])
    #     brownlow[1] = shorten_surname(brownlow[1])
    #     brownlow[2] = shorten_surname(brownlow[2])
        
    #     brownlowdict = {brownlow[0]: 3, brownlow[1]:2, brownlow[2]: 1}
    # else:
    #     brownlowdict = dict()
    
    # brownlowteam1 = list()
    # brownlowteam1.append('Brownlow Votes')
    # for i in range(1, len(team1playerlist)):
    #     if nametransf(team1playerlist[i]) in brownlowdict:
    #         brownlowteam1.append(brownlowdict[nametransf(team1playerlist[i])])
    #     else:
    #         brownlowteam1.append(0)
    # gamestatscol1.append(brownlowteam1)

    # brownlowteam2 = list()
    # brownlowteam2.append('Brownlow Votes')
    # for i in range(1, len(team2playerlist)):
    #     if nametransf(team2playerlist[i]) in brownlowdict:
    #         brownlowteam2.append(brownlowdict[nametransf(team2playerlist[i])])
    #     else:
    #         brownlowteam2.append(0)
    # gamestatscol2.append(brownlowteam2)
    
    
    # Print games where there is an error in Brownlow Votes (i.e. Total brownlow votes does not add up to 1+2+3 = 6)
    # This is a warning mechanism
    # if (sum(brownlowteam1[1:]) + sum(brownlowteam2[1:])) != 6:
    #     print(f'{Year} {Round} {team1} v {team2}: {sum(brownlowteam1[1:]) + sum(brownlowteam2[1:])}')
    #     if brownlow:
    #         print(brownlowdict)
    #     print('\n')
    
    
    return gamestatscol1, gamestatscol2, (Year, Round, team1, team2)



def shorten_surname(name):
    """ Function to change compound surname into semi-initials format to fit the player name data on FootyWire.com """
    
    if '-' in name.split()[1]:
        surname_lst = name.split()[1].split('-')
        new_surname = f'{surname_lst[0][0]}-{surname_lst[1]}'
        return f'{name.split()[0]} {new_surname}'
    
    return name



def getdata(teamplayerstats):
    """ Function which manipulates scraped data and puts them into a suitable list format for further wrangling"""
    
    gamestatsrow = list()
    for row in teamplayerstats:
        record = list()
        cells = row.find_all('td')
        for cell in cells:
            record.append(re.findall(r'>.*<', str(cell))[0].strip('><'))
        gamestatsrow.append(record)

        
    # Add player name    
    gamestatscol = list()
    tmp = list()
    for i in range(len(gamestatsrow)):
        tmp.append(re.findall(r'>.*<',gamestatsrow[i][0])[0].strip('><').split('<')[0])
    gamestatscol.append(tmp)

    
    # Add other stats
    unused_sub = 0
    for i in range(1,16):    # Hardcoded
        tmp = list()

        if i == 1:
            for j in range(len(gamestatsrow)):
                if j == 0:
                    tmp.append(re.findall(r'title=".*"', str(gamestatsrow[0][i]))[0].strip('title=""'))
                elif gamestatsrow[j][i] == 'Unused Substitute':
                    gamestatscol[0] = gamestatscol[0][:-1]
                    unused_sub = 1
                else:
                    tmp.append(float(gamestatsrow[j][i]))

        else:
            n_players_used = len(gamestatsrow)
            if unused_sub:
                n_players_used = len(gamestatsrow)-1

            for j in range(n_players_used):
                if j == 0:
                    tmp.append(re.findall(r'title=".*"', str(gamestatsrow[0][i]))[0].strip('title=""'))
                else:
                    tmp.append(float(gamestatsrow[j][i]))


        gamestatscol.append(tmp)
    
    
    return gamestatscol



def getadvdata(teamplayerstats):
    """ Function which manipulates scraped advanced data and puts them into a suitable list format for further wrangling"""
    
    gamestatsrow = list()
    for row in teamplayerstats:
        record = list()
        cells=row.find_all('td')
        for cell in cells:
            record.append(re.findall(r'>.*<', str(cell))[0].strip('><'))
        gamestatsrow.append(record)
    
    
    # Add player name 
    gamestatscolA = list()
    tmp = list()
    for i in range(len(gamestatsrow)):
        tmp.append(re.findall(r'>.*<',gamestatsrow[i][0])[0].strip('><'))
    gamestatscolA.append(tmp)

    
    # Add other stats
    unused_sub = 0
    for i in range(1,18):    # Hardcoded
        tmp = list()

        if i == 1:
            for j in range(len(gamestatsrow)):
                if j == 0:
                    tmp.append(re.findall(r'title=".*"', str(gamestatsrow[0][i]))[0].strip('title=""'))
                elif gamestatsrow[j][i] == 'Unused Substitute':
                    gamestatscolA[0] = gamestatscolA[0][:-1]
                    unused_sub = 1
                else:
                    tmp.append(float(gamestatsrow[j][i]))

        else:
            n_players_used = len(gamestatsrow)
            if unused_sub:
                n_players_used = len(gamestatsrow)-1

            for j in range(n_players_used):
                if j == 0:
                    tmp.append(re.findall(r'title=".*"', str(gamestatsrow[0][i]))[0].strip('title=""'))
                else:
                    tmp.append(float(gamestatsrow[j][i]))


        gamestatscolA.append(tmp)

        
    return gamestatscolA



def nametransf(name):
    """ For matching full player name format of original data to 'initials + last name' format of Brownlow data """
    
    # Special case: sydney player Josh Kennedy who is recoreded as Josh P. Kennedy 
    if name == 'Josh P. Kennedy':
        return 'J Kennedy'
    
    tmp = name.split(' ')
    first_name = tmp[0]
    last_name = str()
    for i in range(1,len(tmp)):
        last_name += tmp[i]
        last_name += ' '
        
    return f'{first_name[0]} {last_name[:-1]}'



def combine(lst1, lst2):
    """ Function to join the two team's lists into one so we could put into dataframe and save """
    
    out = list()
    for i in range(len(lst1)):
        out.append(list())
        for j in range(len(lst1[i])):
            out[i].append(lst1[i][j])
    for i in range(len(lst1)):
        for j in range(1,len(lst2[i])):
            out[i].append(lst2[i][j])
    return out



def save(lst, metadata, datatype):
    """ Function to save data in form of lists as a dataframe and then a CSV """
    
    DICT = {'O': 'OriginalData'}
    df = pd.DataFrame({lst[0][0]: lst[0][1:len(lst[0])]})
    for i in range(1, len(lst)):
        df.insert(column = lst[i][0], value = lst[i][1:len(lst[0])], loc = i)
    
    if not os.path.exists(f'../future data/raw/{DICT[datatype]}'):
            os.makedirs(f'../future data/raw/{DICT[datatype]}')
    
    df.to_csv(f'../future data/raw/{DICT[datatype]}/{metadata[0]} {metadata[1]} {metadata[2]} v {metadata[3]} ({datatype}).csv', index = False)


# Crawler
base_url = 'https://www.footywire.com/afl/footy/'
years = range(int(sys.argv[1]), int(sys.argv[1])+1)

urllist = list()

for year in years:
    u = f'https://www.footywire.com/afl/footy/ft_match_list?year={year}'
    page = requests.get(u)
    soup = BeautifulSoup(page.text, 'html.parser')
    sections = soup.find_all('a')
    gameurl = list()
    for section in sections:
        if re.search(r'>\d*-\d*<', str(section)):
            gameurl.append(urljoin(base_url, section['href']))
    urllist.append(gameurl)
    
    time.sleep(random.uniform(0.5, 1))

if not os.path.exists(f'../future data/raw'):
    os.makedirs(f'../future data/raw')

try:
    PROCESSED_FILELIST = os.listdir(f'../future data/curated/NormalisedData')
except:
    PROCESSED_FILELIST = []

# Scraper
for year in urllist:
    for game in year:
        test1, test2, metadata = scrape(game)
        
        if test1 != False and test2 != False and metadata != False:
            out = combine(test1, test2)
            save(out, metadata, 'O')
        
        time.sleep(random.uniform(0.5, 1))
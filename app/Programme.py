from flask import Flask
from flask import render_template, url_for, request, redirect

import subprocess
from datetime import date
from Read import read_leaderboard, read_game_by_game_prediction



app = Flask(__name__)


#  HomePage
@app.route("/")
def home():
    return render_template('home.html', date = date.today(), tables=[read_leaderboard().to_html(classes='data', index=False)])


@app.route("/scrape_data")
def scrape_data():
    subprocess.run(f'python crawl_scrape.py {date.today().year}', shell=True, cwd = '../scripts')
    print('Finished Crawling')
    subprocess.run(f'python process_data.py {date.today().year}', shell=True, cwd = '../scripts')

    return render_template('home.html', date = date.today(), tables=[read_leaderboard().to_html(classes='data', index=False)])


@app.route("/run_predictions")
def run_predictions():
    subprocess.run(f'python predict.py {date.today().year}', shell=True, cwd = '../scripts')

    return render_template('home.html', date = date.today(), tables=[read_leaderboard().to_html(classes='data', index=False)])


@app.route("/view_game_by_game_prediction")
def view_game_by_game_prediction():

    return render_template("game_by_game.html", date = date.today(), tables=[read_leaderboard().to_html(classes='data', index=False)], table_list = read_game_by_game_prediction())


@app.route("/view_algo_description")
def view_algo_description():

    return render_template("algo_description.html")

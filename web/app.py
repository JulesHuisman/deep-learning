import numpy as np
import tensorflow as tf

from flask import Flask, send_from_directory, render_template, request, jsonify

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

global sess
global graph

sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)

from web_game import WebGame
from web_deepfour import WebDeepFour, Config, move

net = WebDeepFour(Config(), only_predict=False)
net.load()

app = Flask(__name__,
            static_url_path='',
            static_folder='static',
            template_folder='templates')

@app.route('/play/ai', methods=['POST'])
def ai_plays():
    flat_board = request.json['cells']

    game = WebGame()
    game.board = game.flat_to_board(flat_board)
    position = move(game, net, set_session, sess, graph)
    response = game.try_move(position, player=-1)

    return response

@app.route('/play/human', methods=['POST'])
def human_plays():
    flat_board = request.json['cells']
    position   = request.json['position']

    game = WebGame()
    game.board = game.flat_to_board(flat_board)
    response = game.try_move(position, player=1)

    return response

@app.route('/')
def hello():
    return render_template('app.html')
    
if __name__ == '__main__':
    app.run()
import base64
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import traceback

from MCTS import MctsPlayer, codificaPozitia
from keras.backend import set_session
import logging

from datetime import date

logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
from Env import Env
from chess import Move
from flask import Flask, Response, request, jsonify, render_template
import chess
import chess.svg
from glob import glob
from Config import ConfigProiect
import pprint
from rapidjson import dump
from _collections import deque, defaultdict

conf = tf.ConfigProto(allow_soft_placement=True, device_count={'CPU': 1, 'GPU': 1})
conf.gpu_options.allow_growth = True
conf.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=conf)
graph = tf.get_default_graph()
result = True
app = Flask(__name__)
s = Env()
player = None
imagini = []
lista_fen = []
mutari = []
dir_pattern = None
dictret = {}

class Nod:
    def __init__(self, fen, img, val):
        self.fen = fen
        self.img = img
        self.val = val


def start(Config):
    global player, sess, dir_pattern
    player = get_player(Config)
    rc = ConfigProiect()
    dir_pattern = os.path.join(rc.folder_candidati, rc.template_nume_candidati % "*")
    set_session(sess)
    app.run(debug=True)


def to_svg(s):
    return base64.b64encode(chess.svg.board(board=s.tabla).encode('utf-8')).decode('utf-8')

def to_svg_fen(s):
    s = chess.Board(fen=s)
    return base64.b64encode(chess.svg.board(board=s).encode('utf-8')).decode('utf-8')

@app.route("/")
def hello():
    global dir_pattern
    names = list(sorted(glob(dir_pattern)))
    names = list(map(lambda x: x.split('/')[-1].split('_')[-1].split('-')[-1].split('.')[0], names))
    return render_template('index.html', names=names, start=s.tabla.fen())

@app.route("/getarbore")
def getarbore():
    global dictret, s
    return jsonify(results=dictret, root=to_svg(s))

@app.route("/test")
def test():
    global s
    return render_template('test.html')

@app.route('/getimg')
def getimg():
    fen = str(request.args.get('fen', default=''))
    return to_svg_fen(fen)

@app.route('/getimages')
def getimages():
    global imagini
    return jsonify(results=imagini)

@app.route('/getfens')
def getfens():
    global lista_fen
    print(lista_fen)
    return jsonify(results=lista_fen)

@app.route('/getactions')
def getactions():
    global mutari
    return jsonify(results=mutari)

@app.route("/newgame")
def newgame():
    global s
    s = Env().reseteaza()
    s.tabla.reset()
    response = app.response_class(
        response=s.tabla.fen(),
        status=200
    )
    return response


@app.route("/move_coordinates")
def move_coordinates():
    global player, dictret, s, lista_fen, imagini
    if not s.tabla.is_game_over():
        deUnde = int(request.args.get('deUnde', default=''))
        panaUnde = int(request.args.get('panaUnde', default=''))
        promovare = True if request.args.get('promovare', default='') == 'true' else False
        if chess.Move(deUnde, panaUnde, promotion=chess.QUEEN if promovare else None) in s.tabla.legal_moves:
            mutare = s.tabla.san(chess.Move(deUnde, panaUnde, promotion=chess.QUEEN if promovare else None))
        else:
            print('Mutare ilegala')
            return jsonify(
                response=s.tabla.fen(),
                status=201
            )
        if mutare is not None and mutare != "":
            print("Omul muta: ", mutare)
            s.muta(mutare, san=True)

            lm = list(s.tabla.legal_moves)
            if len(lm) == 1:
                s.muta(lm[0].uci())
                return jsonify(
                    response=s.tabla.fen(),
                    arbore=dictret,
                    status=200
                )

            mutare, fens, vals, dictionar = player.gasesteMutare(s)

            vector_tati = [(Nod(s.tabla.fen(), "data:image/svg+xml;base64," + to_svg_fen(s.tabla.fen()), 0),
                            Nod(s.tabla.fen(), "data:image/svg+xml;base64," + to_svg_fen(s.tabla.fen()), 0))]
            for parentfen, child in dictionar.items():
                for mtr, detaliiMutare in child.detalii.items():
                    if detaliiMutare.nr_viz != 0:   
                        e = Env()
                        e.tabla = chess.Board(parentfen)
                        e.tabla.push(mtr)
                        vector_tati.append(
                            (Nod(e.tabla.fen(), "data:image/svg+xml;base64," + to_svg_fen(e.tabla.fen()),
                                 detaliiMutare.toString()),
                                Nod(parentfen, "data:image/svg+xml;base64," + to_svg_fen(parentfen), child.sum_n)))

            nodes = {}
            for i in vector_tati:
                node, parent_node = i
                nodes[node.fen] = {'img': node.img, 'name': node.val, 'fen': node.fen}
            dictret = []
            for i in vector_tati:
                nodeObj, parentNode = i
                node = nodes[nodeObj.fen]

                # either make the node a new tree or link it to its parent
                if nodeObj.fen == parentNode.fen:
                    # start a new tree in the forest
                    dictret.append(node)
                else:
                    # add new_node as child to parent
                    parent = nodes[parentNode.fen]
                    if not 'children' in parent:
                        # ensure parent has a 'children' field
                        parent['children'] = []
                    children = parent['children']
                    children.append(node)
            print("Computer muta_model ", mutare)
            #player.deboog(s.copiaza())
            lista_fen.append(s.tabla.fen())
            imagini.append(to_svg(s))

            s.muta(mutare)
            return jsonify(
                response=s.tabla.fen(),
                fens=list(map(to_svg_fen, fens)),
                vals=vals,
                arbore=dictret,
                status=200
            )


    print("GAME IS OVER")
    return jsonify(
            response="joc_din_pgn over",
            status=200
        )


@app.route("/detalii")
def detalii():
    global dir_pattern
    names = list(sorted(glob(dir_pattern)))
    names = list(map(lambda x: x.split('/')[-1].split('_')[-1].split('-')[-1].split('.')[0], names))
    return render_template('detalii.html', names=names)

@app.route("/selfplay")
def selfplay():
    global player, s, imagini, lista_fen, mutari, dir_pattern
    names = list(sorted(glob(dir_pattern)))
    names = list(map(lambda x: x.split('/')[-1].split('_')[-1].split('-')[-1].split('.')[0], names))
    imagini.append(to_svg(s))
    while not s.tabla.is_game_over(claim_draw=True):
        action, fens, vals, arbore = player.gasesteMutare(s.copiaza())
        s.muta(action)
        imagini.append(to_svg(s))
        lista_fen.append(fens)
        mutari.append(s.tabla.fen())
    return render_template('selfplay.html', names=names)


def get_player(config):
    from ModelAC import ActorCritic
    from lib.model_helper import incarca_parametrii_model_bun
    global sess
    model = ActorCritic(config, sess)
    if not incarca_parametrii_model_bun(model):
        raise RuntimeError("Best model not found!")
    return MctsPlayer(config, model.get_pipes(128), for_web=True)

import os
import rapidjson
from datetime import datetime
from glob import glob
from logging import getLogger

import chess
import chess.pgn
import pyperclip
from Config import ConfigProiect
import numpy as np

logger = getLogger(__name__)


def pretty_print(env, colors):
    new_pgn = open("test3.pgn", "at")
    game = chess.pgn.Game.from_board(env.tabla)
    game.headers["Result"] = env.rezultat
    game.headers["White"], game.headers["Black"] = colors
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    new_pgn.write(str(game) + "\n\n")
    new_pgn.close()
    pyperclip.copy(env.tabla.fen())


def find_pgn_files(directory, pattern='*.pgn'):
    dir_pattern = os.path.join(directory, pattern)
    files = list(sorted(glob(dir_pattern)))
    return files


def get_game_data_filenames(rc: ConfigProiect, spvz=False):
    if spvz:
        pattern = os.path.join(rc.fld_spvz_train_data, rc.fld_date_pattern % "*")
    else:
        pattern = os.path.join(rc.fld_date, rc.fld_date_pattern % "*")
    files = list(sorted(glob(pattern)))
    return files


def get_next_generation_model_dirs(rc: ConfigProiect):
    dir_pattern = os.path.join(rc.folder_candidati, rc.template_nume_candidati % "*")
    dirs = list(sorted(glob(dir_pattern)))
    return dirs


def write_game_data_to_file(path, data):
    try:
        with open(path, "wt") as f:
            rapidjson.dump(data, f)
        #print('--------------------------------------------date salvate')
    except Exception as e:
        print('--------WRITE DATA EXCEPTION------------ ' + str(e))
        #print(data)


def read_game_data_from_file(path):
    try:
        with open(path, "rt") as f:
            x = rapidjson.load(f)
            return x
    except Exception as e:
        print(str(e) + path)
        return None

def decomprima_mutare(mutare):
    m = np.zeros(2012)
    for x in mutare:
        m[x[0]] = x[1]
    return list(m)

def comprima_mutare(mutare):
    idx = np.nonzero(mutare)[0]
    if len(idx) == 0:
        return [[0, 0]]
    idx = list(map(int, idx))
    val = list(np.array(mutare)[idx])
    return list(map(list, list(zip(idx, val))))

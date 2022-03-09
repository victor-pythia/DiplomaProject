import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from datetime import datetime
from logging import getLogger
from threading import Thread
from time import time
from tqdm import tqdm
from keras.backend import set_session
import chess.pgn
import sys

sys.setrecursionlimit(1000000000)
from MCTS import MctsPlayer
from Config import Config
from Env import Env, Winner
from lib.data_helper import write_game_data_to_file, find_pgn_files, comprima_mutare, decomprima_mutare
import tensorflow as tf
from time import sleep
from shutil import move

logger = getLogger(__name__)

TAG_REGEX = re.compile(r"^\[([A-Za-z0-9_]+)\s+\"(.*)\"\]\s*$")

result = False
while not result:
    try:
        conf = tf.ConfigProto(allow_soft_placement=True, device_count={'CPU': 1, 'GPU': 1})
        conf.gpu_options.allow_growth = True
        conf.gpu_options.per_process_gpu_memory_fraction = 0.99
        sess = tf.Session(config=conf)
        graph = tf.get_default_graph()
        result = True
    except:
        sleep(1)


def start(config: Config):
    return ConvertorSupervizat(config).start()


class ConvertorSupervizat:
    def __init__(self, config: Config):
        self.config = config
        self.buffer = []

    def start(self):
        self.buffer = []
        self.idx = 0
        games = self.transforma_mutarile()
        with ProcessPoolExecutor(max_workers=28) as executor:
            for res in tqdm(as_completed(
                    [executor.submit(obtine_reprezentari, self.config, game) for game in games])):
                self.idx += 1
                env, data = res.result()
                self.salveaza_datele(data)

        if len(self.buffer) > 0:
            self.flush_buffer()

    def transforma_mutarile(self):
        fisiere_pgn = find_pgn_files(self.config.confProiect.fld_date_spvz_hdd)
        jocuri = []
        with ProcessPoolExecutor(max_workers=2) as executor:
            for res in as_completed([executor.submit(preia_jocuri_din_fisier,
                                                     fisier,
                                                     self.config.confProiect.fld_date_spvz_done,
                                                     self.config.conf_joc_spvz.interval) for fisier in fisiere_pgn]):
                x = res.result()
                jocuri.extend(x)
        print("Date citite")
        return jocuri

    def salveaza_datele(self, data):
        self.buffer += data
        if self.idx % self.config.conf_joc_spvz.sl_nb_game_in_file == 0:
            self.flush_buffer()

    def flush_buffer(self):
        rc = self.config.confProiect
        id_joc = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        cale = os.path.join(rc.fld_spvz_train_data, rc.fld_date_spvz_pattern % id_joc)
        #logger.info("Salvez datele in {}".format(cale))
        thread = Thread(target=write_game_data_to_file, args=(cale, self.buffer))
        thread.start()
        self.buffer = []


def citeste_joc(nume_pgn, offset):
    with open(nume_pgn, errors='ignore') as pgn_file:
        try:
            pgn_file.seek(offset)
            x = chess.pgn.read_game(pgn_file)
        except:
            pass
    return x


def preia_jocuri_din_fisier(nume_fisier, dest, interval):
    logger.info(f"Deschid {nume_fisier}")
    pgn = open(nume_fisier, errors='ignore')
    offsets = []
    jocuri = []
    idx = 0
    while True:
        if idx <= interval[0]:
            idx += 1
            continue
        offset = pgn.tell()
        headers = chess.pgn.read_headers(pgn)
        if headers is None or idx > interval[1]:
            break
        # if headers["Variant"] == "suicide" and int(headers["PlyCount"]) > 5:
        #     offsets.append(offset)
        #     idx += 1
        offsets.append(offset)
        idx += 1
    pgn.close()


    print(f"{len(offsets)} jocuri in {nume_fisier}")
    with ProcessPoolExecutor(28) as executor:
        for res in tqdm(as_completed(
                [executor.submit(
                    citeste_joc, nume_fisier, offset) for idx, offset in enumerate(offsets)
                 ])):
            joc = res.result()
            jocuri.append(joc)
    move(nume_fisier, dest)
    return jocuri


def pondereaza(config, elo):
    return min(1, max(0, elo - config.conf_joc_spvz.min_elo_policy) / config.conf_joc_spvz.max_elo_policy)


def obtine_reprezentari(config, joc_din_pgn) -> (Env, list):
    """
    Joaca meciuri din baza de date si le salveaza in formatul necesar pentru antrenare
    """
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        env = Env().reseteaza()
        jucator_alb = MctsPlayer(config, supervizat=True)
        jucator_negru = MctsPlayer(config, supervizat=True)
        rezultat = joc_din_pgn.headers["Result"]
        if joc_din_pgn.headers["WhiteElo"] == '?':
            joc_din_pgn.headers["WhiteElo"] = '0'
        if joc_din_pgn.headers["BlackElo"] == '?':
            joc_din_pgn.headers["BlackElo"] = '0'
        elo_alb, elo_negru = int(joc_din_pgn.headers["WhiteElo"]), int(joc_din_pgn.headers["BlackElo"])
        pondere_alb = pondereaza(config, elo_alb)
        pondere_negru = pondereaza(config, elo_negru)

        mutari = []
        aaa = 0
        while not joc_din_pgn.is_end():
            aaa += 1
            joc_din_pgn = joc_din_pgn.variation(0)
            mutari.append(joc_din_pgn.move.uci())
        k = 0
        while not env.sfarsit and k < len(mutari):
            if env.muta_alb:
                mutare = jucator_alb.mutare_supervizat(env.vezi_fen, mutari[k], pondere=pondere_alb)  # ignore=True
            else:
                mutare = jucator_negru.mutare_supervizat(env.vezi_fen, mutari[k], pondere=pondere_negru)  # ignore=True
            env.muta(mutare, False)
            k += 1
        if not env.tabla.is_game_over(claim_draw=True) and rezultat != '1/2-1/2':
            env.cedat = True
        if rezultat == '1-0':
            env.castigator = Winner.white
            black_win = -1
        elif rezultat == '0-1':
            env.castigator = Winner.black
            black_win = 1
        else:
            env.castigator = Winner.draw
            black_win = 0


        data = []
        for i in range(len(jucator_alb.mutari)):
            jucator_alb.mutari[i][0] = comprima_mutare(jucator_alb.mutari[i][0])
            data.append(jucator_alb.mutari[i] + [-black_win])
            if i < len(jucator_negru.mutari):
                jucator_negru.mutari[i][0] = comprima_mutare(jucator_negru.mutari[i][0])
                data.append(jucator_negru.mutari[i] + [black_win])
        # jucator_negru.adauga_castigator(black_win)
        # jucator_alb.adauga_castigator(-black_win)


        data = [[data, env.stivaFen]]     # pt istorie
        return env, data

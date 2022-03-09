import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from ModelAC import ActorCritic
from MCTS import MctsPlayer
from Env import Env, Winner
from multiprocessing import Manager
from collections import deque
from threading import Thread
from time import time, sleep
from concurrent.futures import ProcessPoolExecutor, as_completed
from Config import Config
from lib.data_helper import get_game_data_filenames, write_game_data_to_file, pretty_print, comprima_mutare
from lib.model_helper import incarca_parametrii_model_bun, salveaza_ca_cel_mai_bun_model, testeaza_schimbari
from lib.tf_util import set_session_config

from keras.backend import set_session
import tensorflow as tf
from datetime import datetime
from logging import getLogger

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logger = getLogger(__name__)

result = False
while not result:
    try:
        conf = tf.ConfigProto(allow_soft_placement=True, device_count={'CPU': 28, 'GPU': 1})
        conf.gpu_options.allow_growth = True
        conf.gpu_options.per_process_gpu_memory_fraction = 0.99
        sess = tf.Session(config=conf)
        graph = tf.get_default_graph()
        result = True
    except:
        sleep(1)


def start(config: Config):
    return SelfPlayWorker(config).start()


class SelfPlayWorker:
    def __init__(self, config):
        self.config = config
        self.model_curent = self.incarca_model()
        self.m = Manager()
        self.cur_pipes = self.m.list([self.model_curent.get_pipes(self.config.conf_joc.search_threads)
                                      for _ in range(self.config.conf_joc.max_processes)])
        self.buffer = []

    def incarca_model(self):
        model = ActorCritic(self.config, sess)
        if self.config.opts.new or not incarca_parametrii_model_bun(model):
            model.creeaza()
            salveaza_ca_cel_mai_bun_model(model)
        return model

    def start(self):
        self.buffer = []
        futures = deque()
        with ProcessPoolExecutor(max_workers=self.config.conf_joc.max_processes) as executor:
            for idx_joc in range(self.config.conf_joc.max_processes * 2):
                futures.append(
                    executor.submit(obtine_reprezentari,
                                    self.config,
                                    cur=self.cur_pipes))
            idx_joc = 0

            while True:
                idx_joc += 1
                env, data, durata = futures.popleft().result()
                logger.info(f"joc {idx_joc:3} durata={durata:5.1f}s "
                            f"{env.nrMutari:3} mutari {env.castigator:12} "
                            f"{'oponentul s-a dat batut ' if env.cedat else '          '}")
                self.buffer += data
                if (idx_joc % self.config.conf_joc_spvz.nb_game_in_file) == 0:
                    self.flush_buffer()
                    testeaza_schimbari(self.model_curent)
                futures.append(executor.submit(obtine_reprezentari, self.config, cur=self.cur_pipes))

    def flush_buffer(self):
        rc = self.config.confProiect
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.fld_date, rc.fld_date_pattern % game_id)
        logger.info(f"salveaza datele in {path}")
        print(f'Date salvate in {path}')
        thread = Thread(target=write_game_data_to_file, args=(path, self.buffer))
        thread.start()
        self.buffer = []


def obtine_reprezentari(config, cur):
    """
        Modelul muta pentru ambii jucatori, retine mutarile si le transforma in reprezentari.
    """
    global sess, graph
    t = time()
    with graph.as_default():
        set_session(sess)
        pipes = cur.pop()  # borrow
        env = Env().reseteaza(pozitie_aleatoare=False, nr_aleator=7)
        alb = MctsPlayer(config, pipes=pipes)
        negru = MctsPlayer(config, pipes=pipes)
        while not env.sfarsit:
            if env.muta_alb:
                mutare = alb.gasesteMutare(env)
            else:
                mutare = negru.gasesteMutare(env)
            env.muta(mutare)
            if env.nrMutari >= config.conf_joc.max_game_length:
                env.analizaObiectiva()
        if env.castigator == Winner.white:
            black_win = -1
        elif env.castigator == Winner.black:
            black_win = 1
        else:
            black_win = 0

        # negru.adauga_castigator(black_win)
        # alb.adauga_castigator(-black_win)

        reprezentari = []
        for i in range(len(alb.mutari)):
            alb.mutari[i][0] = comprima_mutare(alb.mutari[i][0])
            reprezentari.append(alb.mutari[i] + [-black_win])
            if i < len(negru.mutari):
                negru.mutari[i][0] = comprima_mutare(negru.mutari[i][0])
                reprezentari.append(negru.mutari[i] + [black_win])

        reprezentari = [[reprezentari, env.stivaFen]]   # pt istorie
        cur.append(pipes)
        return env, reprezentari, time() - t

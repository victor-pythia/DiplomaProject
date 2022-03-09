import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger
from multiprocessing import Manager
from time import sleep
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
from ModelAC import ActorCritic
from MCTS import MctsPlayer
from Config import Config
from Env import Env, Winner
from lib.data_helper import get_next_generation_model_dirs, pretty_print
from lib.model_helper import salveaza_ca_cel_mai_bun_model, incarca_parametrii_model_bun
import tensorflow as tf
from csv import writer

logger = getLogger(__name__)

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
    return EvaluateWorker(config).start()


class EvaluateWorker:
    """
    Worker pentru evaluare. Se vor juca meciuri intre generatia noua ( ultimul model antrenat ) si modelul declarat
    ca fiind cel mai bun.
    """

    def __init__(self, config: Config):
        self.config = config
        self.conf_joc = config.conf_joc
        self.model_curent = self.incarca_cel_mai_bun_model()
        self.m = Manager()
        self.cur_pipes = self.m.list([self.model_curent.get_pipes(self.conf_joc.search_threads) for _ in
                                      range(self.conf_joc.max_processes)])

    def start(self):
        while True:
            candidat, folder_model = self.incarca_candidat()
            logger.debug(f"Evaluez candidatul: {folder_model}")
            candidat_ok = self.evalueaza(candidat)
            if candidat_ok:
                logger.debug(f"Candidatul devinde cel mai bun model: {folder_model}")
                salveaza_ca_cel_mai_bun_model(candidat)
                self.model_curent = candidat
                self.muta_model(folder_model)
            self.evalueaza_vs_stockfish(candidat=candidat, nivel=self.conf_joc.nivelStockfish)

    def evalueaza_vs_stockfish(self, candidat, nivel=1):
        pipes_candidat = self.m.list(
            [candidat.get_pipes(self.conf_joc.search_threads) for _ in range(self.conf_joc.max_processes)])
        futures = []

        with ProcessPoolExecutor(max_workers=self.conf_joc.max_processes) as executor:
            for game_idx in range(self.config.eval.game_num):
                fut = executor.submit(joaca_vs_stockfish,
                                      self.config,
                                      pipes_candidat,
                                      (game_idx % 2 == 0),
                                      self.conf_joc.nivelStockfish)
                futures.append(fut)
            rezultate = []

            for fut in as_completed(futures):
                # 1: castiga candidatul, 0: pierde candidatul, 0.5: remiza

                scor_candidat, env, jucator_alb = fut.result()

                rezultate.append(scor_candidat)
                rata_castig = sum(rezultate) / len(rezultate)
                game_idx = len(rezultate)
                # logger.debug(f"Joc nr {game_idx:3}: scor_candidat={scor_candidat:.1f} jucand ca"
                #              f" {'negru' if jucator_alb else 'alb'} "
                #              f"{'prin cedare ' if env.cedat else env.tabla.result(claim_draw=True)}"
                #              f" rata_castig={rata_castig * 100:5.1f  }% "
                #              f"{env.nrMutari:3} mutari")
                print(scor_candidat)

                jucatori = ("stockfish", "candidat")
                if not jucator_alb:
                    jucatori = reversed(jucatori)
                pretty_print(env, jucatori)

        rata_castig = sum(rezultate) / len(rezultate)
        with open('Rata_Stockfish.csv', 'a+') as f:
            csv_writer = writer(f)
            csv_writer.writerow(rata_castig)

        logger.debug(f"Rata de castig {rata_castig * 100:.1f}%")
        return rata_castig >= self.config.eval.replace_rate

    def evalueaza(self, candidat):
        pipes_candidat = self.m.list(
            [candidat.get_pipes(self.conf_joc.search_threads) for _ in range(self.conf_joc.max_processes)])
        futures = []
        with ProcessPoolExecutor(max_workers=self.conf_joc.max_processes) as executor:
            for index_joc in range(self.config.eval.game_num):
                fut = executor.submit(joaca, self.config,
                                      cur=self.cur_pipes,
                                      ng=pipes_candidat,
                                      jucator_alb=(index_joc % 2 == 0))
                futures.append(fut)

            rezultate = []
            for fut in as_completed(futures):
                # 1: castiga candidatul, 0: pierde candidatul, 0.5: remiza
                scor_candidat, env, jucator_alb = fut.result()
                rezultate.append(scor_candidat)
                rata_castig = sum(rezultate) / len(rezultate)
                index_joc = len(rezultate)
                logger.debug(f"Joc nr {index_joc:3}: scor_candidat={scor_candidat:.1f} jucand ca"
                             f" {'negru' if jucator_alb else 'alb'} "
                             f"{'prin cedare ' if env.cedat else env.tabla.result(claim_draw=True)}"
                             f" rata_castig={rata_castig * 100:5.1f}% "
                             f"{env.nrMutari} mutari")

                jucatori = ("model_curent", "candidat")
                if not jucator_alb:
                    jucatori = reversed(jucatori)
                pretty_print(env, jucatori)

                if len(rezultate) - sum(rezultate) >= self.config.eval.game_num * (1 - self.config.eval.replace_rate):
                    logger.debug(f"Nr de infrangeri {rezultate.count(0)}, continuare inutila")
                    executor.shutdown(wait=False)
                    return False
                if sum(rezultate) >= self.config.eval.game_num * self.config.eval.replace_rate:
                    logger.debug(
                        f"Nr de castiguri {rezultate.count(1)}, deci candidatul a devenit cel mai bun model")
                    executor.shutdown(wait=False)
                    return True

        rata_castig = sum(rezultate) / len(rezultate)
        logger.debug(f"Rata de castig {rata_castig * 100:.1f}%")
        return rata_castig >= self.config.eval.replace_rate

    def muta_model(self, model_dir):
        rc = self.config.confProiect
        folder_nou = os.path.join(rc.folder_candidati, "copies", model_dir)
        os.rename(model_dir, folder_nou)

    def incarca_cel_mai_bun_model(self):
        model = ActorCritic(self.config, sess)
        incarca_parametrii_model_bun(model)
        return model

    def incarca_candidat(self):
        rc = self.config.confProiect
        while True:
            dirs = get_next_generation_model_dirs(self.config.confProiect)
            if dirs:
                break
            logger.info("Nu exista candidati")
            sleep(60)
        director_model = dirs[-1] if self.config.eval.evaluate_latest_first else dirs[0]
        config_path = os.path.join(director_model, rc.candidat_config)
        weight_path = os.path.join(director_model, rc.candidat_parametrii)
        model = ActorCritic(self.config, sess)
        model.incarca(config_path, weight_path)
        return model, director_model.split('/')[-1]


def joaca(config, cur, ng, jucator_alb: bool) -> (float, Env, bool):
    pipes_curent = cur.pop()
    pipes_candidat = ng.pop()
    env = Env().reseteaza(pozitie_aleatoare=False)

    jucator_curent = MctsPlayer(config, pipes=pipes_curent, conf_joc=config.eval.conf_joc)
    jucator_candidat = MctsPlayer(config, pipes=pipes_candidat, conf_joc=config.eval.conf_joc)
    if jucator_alb:
        alb, negru = jucator_curent, jucator_candidat
    else:
        alb, negru = jucator_candidat, jucator_curent

    while not env.sfarsit:
        if env.muta_alb:
            action = alb.gasesteMutare(env)
        else:
            action = negru.gasesteMutare(env)
        env.muta(action)
        if env.nrMutari >= config.eval.max_game_length:
            env.analizaObiectiva()

    if env.castigator == Winner.draw:
        scor_candidat = 0.5
    elif env.a_castigat_alb == jucator_alb:
        scor_candidat = 0
    else:
        scor_candidat = 1
    cur.append(pipes_curent)
    ng.append(pipes_candidat)
    return scor_candidat, env, jucator_alb


def joaca_vs_stockfish(config, cand, jucator_alb, nivel):
    from chess.engine import SimpleEngine, Limit
    cand_pipes = cand.pop()
    env = Env().reseteaza()

    candidat = MctsPlayer(config, pipes=cand_pipes, conf_joc=config.eval.conf_joc)
    motor = SimpleEngine.popen_uci("/usr/games/stockfish")
    while not env.sfarsit:
        if env.muta_alb and jucator_alb:
            action = candidat.gasesteMutare(env)
            env.muta(action)
        elif env.muta_alb and not jucator_alb:
            action = motor.play(env.tabla, Limit(time=0.001, depth=nivel))
            env.muta(action.move.uci())
        elif not env.muta_alb and jucator_alb:
            action = motor.play(env.tabla, Limit(time=0.001, depth=nivel))
            env.muta(action.move.uci())
        else:
            action = candidat.gasesteMutare(env)
            env.muta(action)

        if env.nrMutari >= config.eval.max_game_length:
            #env.analizaObiectiva()
            env.castigator = Winner.draw
            break

    if env.castigator == Winner.draw:
        scor_candidat = 0.5
    elif env.a_castigat_alb == jucator_alb:
        scor_candidat = 1
    else:
        scor_candidat = 0
    cand.append(cand_pipes)
    motor.quit()
    return scor_candidat, env, jucator_alb

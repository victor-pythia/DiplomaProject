import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from time import sleep
from random import shuffle
import tensorflow as tf
import numpy as np
from tensorflow.python.util import deprecation
from Env import Env, reprezenetari_plus_detalii

deprecation._PRINT_DEPRECATION_WARNINGS = False
from ModelAC import ActorCritic
from Config import Config
from Env import planuriAZ, muta_negru, evaluare_obiectiva, intoarceTabla, fen_catre_planuri, planuri_detaliu
from lib.data_helper import get_game_data_filenames, read_game_data_from_file, get_next_generation_model_dirs, decomprima_mutare
from lib.model_helper import incarca_parametrii_model_bun

from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.utils import Sequence
from shutil import move
import sys
from memory_profiler import profile

logger = getLogger(__name__)

result = False
while not result:
    try:
        conf = tf.ConfigProto(allow_soft_placement=True, device_count={'CPU': 1, 'GPU': 1})
        conf.gpu_options.allow_growth = True
        conf.gpu_options.per_process_gpu_memory_fraction = 0.95
        sess = tf.Session(config=conf)
        graph = tf.get_default_graph()
        result = True
    except:
        sleep(1)

cf = None

def start(config: Config):
    global cf
    cf = config
    return Antrenor(config).start()


class GeneratorDate(Sequence):
    def __init__(self, filenames, batch_size=1):
        self.dataset_filenames = filenames
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.dataset_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        dt = deque(), deque(), deque()
        for date in self.dataset_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]:
            reprezentari, strategii, valori = incarca_datele_din(date)
            dt[0].append(reprezentari)
            dt[1].append(strategii)
            dt[2].append(valori)
        reprezentari, strategii, valori = np.asarray(dt[0]), np.asarray(dt[1]), np.asarray(dt[2])
        return np.squeeze(reprezentari, axis=0), [np.squeeze(strategii, axis=0), np.squeeze(valori, axis=0)]


class Antrenor:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.dataset = deque(), deque(), deque()
        self.executor = ProcessPoolExecutor(max_workers=config.conf_antr.cleaning_processes)


    def start(self):
        self.model = self.incarca_model()
        self.antreneaza()

    def antreneaza(self):
        global sess, conf, graph
        result = False
        while not result:
            try:
                if not sess or not conf or not graph:
                    conf = tf.ConfigProto(allow_soft_placement=True, device_count={'CPU': 1, 'GPU': 1})
                    conf.gpu_options.allow_growth = True
                    conf.gpu_options.per_process_gpu_memory_fraction = 0.99
                    sess = tf.Session(config=conf)
                    graph = tf.get_default_graph()
                    result = True
                else:
                    result = True
            except:
                sleep(1)
        mbc = 125
        self.copileaza_model()
        total_steps = self.config.conf_antr.start_total_steps
        self.filenames_all = get_game_data_filenames(self.config.confProiect)
        for i in range(0, len(self.filenames_all) // mbc, mbc):
            self.filenames = self.filenames_all[i:i+mbc]
            shuffle(self.filenames)
            self.preia_datele()
            steps = self.antreneaza_epoci(self.config.conf_antr.epoch_to_checkpoint)
            total_steps += steps
            self.salveaza_model_curent()
            a, b, c = self.dataset
            while len(a) > self.config.conf_antr.dataset_size / 2:
                a.popleft()
                b.popleft()
                c.popleft()
            for fl in self.filenames:
                move(fl,  '/media/lilmarco/Seagate_HDD/LICENTA_DATE/Done_train')
                print('Am mutat datele')
            self.executor.shutdown(wait=False)
            del self.executor, a, b, c, self.dataset
            self.dataset = deque(), deque(), deque()
            self.executor = ProcessPoolExecutor(max_workers=self.config.conf_antr.cleaning_processes)


    def antreneaza_epoci(self, nr_epoci):
        tc = self.config.conf_antr
        reprezentari, strategii, valori = self.uneste_DB()
        tensorboard_cb = TensorBoard(log_dir="./logs", batch_size=tc.batch_size, histogram_freq=1)
        with open('./data/current_epoch.txt', 'r') as f:
            ie = int(f.readline())
        self.model.model.fit(reprezentari,
                             [strategii, valori],
                             batch_size=tc.batch_size,
                             epochs=ie+nr_epoci,
                             initial_epoch=ie,
                             shuffle=True,
                             validation_split=0.02,
                             callbacks=[tensorboard_cb])
        with open('./data/current_epoch.txt', 'w') as f:
            f.write(str(ie+nr_epoci))
        # fl = list(self.filenames)
        # self.model.model.fit_generator(
        #     generator=GeneratorDate(fl[len(self.filenames)//10:], 1),
        #     validation_data=GeneratorDate(fl[:len(fl)//10], 1),
        #     use_multiprocessing=True,
        #     workers=16
        # )
        steps = (reprezentari.shape[0] // tc.batch_size) * nr_epoci
        return steps

    def copileaza_model(self):
        opt = Adam(lr=0.000005)
        losses = ['kullback_leibler_divergence', 'mean_squared_error']
        self.model.model.compile(optimizer=opt, loss=losses, loss_weights=self.config.conf_antr.loss_weights)
        self.model.model.summary()


    def salveaza_model_curent(self):
        rc = self.config.confProiect
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        model_dir = os.path.join(rc.folder_candidati, rc.template_nume_candidati % model_id)
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, rc.candidat_config)
        weight_path = os.path.join(model_dir, rc.candidat_parametrii)
        self.model.salveaza(config_path, weight_path)

    def preia_datele(self):
        futures = deque()
        flc = deque(self.filenames)
        for _ in range(self.config.conf_antr.cleaning_processes):
            if len(flc) == 0:
                break
            filename = flc.popleft()
            logger.debug(f"Incarc datele din {filename}")
            futures.append(self.executor.submit(incarca_datele_din, filename))
        while futures and len(self.dataset[0]) < self.config.conf_antr.dataset_size:
            if futures[0] is not None:
                for x, y in zip(self.dataset, futures.popleft().result()):
                    x.extend(y)
                    del y
            if len(flc) > 0:
                filename = flc.popleft()
                logger.debug(f"Incarc datele din {filename}")
                futures.append(self.executor.submit(incarca_datele_din, filename))

    def uneste_DB(self):
        return [np.asarray(self.dataset[i], dtype=np.float32) for i in range(len(self.dataset))]

    def incarca_model(self):
        global sess
        model = ActorCritic(self.config, sess)
        rc = self.config.confProiect

        dirs = get_next_generation_model_dirs(rc)
        if not dirs:
            logger.debug("Incarc cel mai bun model")
            if not incarca_parametrii_model_bun(model):
                raise RuntimeError("Cel mai bun model nu poate fi incarcat")
        else:
            latest_dir = dirs[-1]
            logger.debug("Incarc ultimul candidat")
            config_path = os.path.join(latest_dir, rc.candidat_config)
            weight_path = os.path.join(latest_dir, rc.candidat_parametrii)
            model.incarca(config_path, weight_path)
        return model


def incarca_datele_din(filename):
    return converteste_in_reprezenetari(read_game_data_from_file(filename))


def converteste_in_planuri(stivaFen):
    planuri = []
    reprezentari = list(map(lambda x: fen_catre_planuri(intoarceTabla(x, muta_negru(x))), stivaFen))
    dq = [np.asarray([np.full((8, 8), 0) for _ in range(12)]) for _ in range(3)]
    for i in range(len(stivaFen)):
        dq.append(reprezentari[i])
        det_plane = planuri_detaliu(stivaFen[i])
        planuri.append(np.vstack((det_plane, np.vstack(dq))))
        del dq[0]
    return planuri


def converteste_in_reprezenetari(data):     # cu istorie
    reprezentari = []
    strategii = []
    valori = []
    for joc in data:
        stivaFen = joc[1]
        planuri = converteste_in_planuri(stivaFen)
        for idx, (strategie, valoare) in enumerate(joc[0]):
            strategie = decomprima_mutare(strategie)
            if muta_negru(stivaFen[idx]):
                strategie = Config.inv_strategie(strategie)
            reprezentari.append(planuri[idx])
            strategii.append(strategie)
            valori.append(valoare)
    return np.asarray(reprezentari, dtype=np.float32),\
           np.asarray(strategii, dtype=np.float32),\
           np.asarray(valori, dtype=np.float32)


# def converteste_in_reprezenetari(data):     # fara istorie
#     reprezentari = []
#     strategii = []
#     valori = []
#     for state_fen, policy, value in data:
#         state_planes = reprezenetari_plus_detalii(state_fen)
#         if muta_negru(state_fen):
#             policy = Config.inv_strategie(policy)
#         move_number = int(state_fen.split(' ')[5])
#         value_certainty = min(5, move_number) / 5  # reduces the noise of the opening... plz train faster
#         sl_value = value * value_certainty #+ naive_evaluation(state_fen, False) * (1 - value_certainty)
#
#         reprezentari.append(state_planes)
#         strategii.append(policy)
#         valori.append(sl_value)
#     return np.asarray(reprezentari, dtype=np.float32), np.asarray(strategii, dtype=np.float32), np.asarray(
#         valori,
#             dtype=np.float32)

# def converteste_in_reprezenetari(data):   # cu istorie
#     reprezentari = []
#     strategii = []
#     valori = []
#     for joc in data:
#         print(joc)
#         fen = joc[1]
#         planuri = fen_catre_planuri(fen)
#         for idx, (strategie, valoare) in enumerate(joc[0]):
#             if muta_negru(fen[idx]):
#                 strategie = Config.inv_strategie(strategie)
#             reprezentari.append(planuri[idx])
#             strategii.append(strategie)
#             valori.append(valoare)
#     return np.asarray(reprezentari, dtype=np.float32),\
#            np.asarray(strategii, dtype=np.float32),\
#            np.asarray(valori, dtype=np.float32)


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import getLogger
from threading import Lock

import chess
import numpy as np

from Config import Config
from Env import Env, Winner, planuriAZ, evaluare_obiectiva, intoarceTabla

logger = getLogger(__name__)

class Nod:
    def __init__(self):
        self.sum_n = 0
        self.fen = None
        self.detalii = defaultdict(DetaliiNod)


class DetaliiNod:
    """
        Retine valorile nodurilor
        :ivar int nr_viz: De cate ori a fost considerata aceasta actiune
        :ivar float cumul: De fiecare data cand unul din fii acestiu nod este vizitat, se acumuleaza o valoare in aceasta
            variabila (cea prezisa de retea). Folosind virtual loss, putem muta_model cautarea mai mult inspre explorare.
        :ivar float prob_medie: nr_viz / cumul
        :ivar float prob_retea: probabilitatea prezisa de retea
    """

    def __init__(self):
        self.prob_retea = 0
        self.nr_viz = 0
        self.cumul = 0
        self.prob_medie = 0


    def __repr__(self):
        return 'prob_retea: {}, prob_medie: {}, cumul: {}, nr_viz: {} '.format(self.prob_retea, self.prob_medie, self.cumul, self.nr_viz)

    def toList(self):
        return [self.prob_retea, self.prob_medie, self.cumul, self.nr_viz]

    def toString(self):
        return f"{self.prob_medie:.2f}, {self.prob_retea:.2f}, {self.cumul:.2f}, {self.nr_viz}"


class MctsPlayer:
    """
    Motorul de sah. Joaca folosind un arbore de decizie MonteCarlo, luand decizii pe baza predictiilor de actiune si
    valoarea stadiului de joc oferite de reteaua neuronala adanca. Reteaua este conectata prin 'pipes'
    """

    # dot = False
    def __init__(self, config: Config, pipes=None, conf_joc=None, supervizat=False, for_web=False):
        self.mutari = []
        self.arbore = defaultdict(Nod)
        self.config = config
        self.conf_joc = conf_joc or self.config.conf_joc
        self.nr_etichete = config.nr_etichete
        self.etichete = config.etichete  # a1a2, a1a3 ...
        self.san_catre_nr = {chess.Move.from_uci(move): i for move, i in
                             zip(self.etichete, range(self.nr_etichete))}  # move(a1a2) : 1 ...
        if supervizat:
            return
        self.pipe_pool = pipes
        self.node_lock = defaultdict(Lock)
        self.for_web = for_web

    def reset(self):
        """
        Reseteaza arborele de decizie
        """
        self.arbore = defaultdict(Nod)

    def deboog(self, env):
        print('naive eval: ', evaluare_obiectiva(env.tabla.fen(), True))

        state = codificaPozitia(env)
        my_visit_stats = self.arbore[state]
        stats = []
        for action, a_s in my_visit_stats.detalii.items():
            moi = self.san_catre_nr[action]
            stats.append(np.asarray([a_s.nr_viz, a_s.cumul, a_s.prob_medie, a_s.prob_retea, moi]))
        stats = np.asarray(stats)
        a = stats[stats[:, 0].argsort()[::-1]]

        for s in a:
            print(f'{self.etichete[int(s[4])]:5}: '
                  f'nr_viz: {s[0]:3.0f} '
                  f'cumul: {s[1]:7.3f} '
                  f'prob_medie: {s[2]:7.3f} '
                  f'prob_retea: {s[3]:7.5f}')

    def gasesteMutare(self, env, can_stop=True) -> str:
        """
        Alege cea mai buna decizie si o returneaza ca string
        """
        self.reset()
        # Generez arborii de cautare explorand/exploatand noduri
        if self.for_web:
            ret = self.cauta_toate_mutarile(env.copiaza())
            root_values = list(np.array(ret)[:3, 0])
            valoarePozitie = ret[0][0]
            fens = list(np.array(ret)[:3, 1])
        else:
            valoarePozitie, naked_value = self.cauta_toate_mutarile(env.copiaza())
        strategie = self.calc_strategie(env.copiaza())
        strategieFinala = int(np.random.choice(range(self.nr_etichete), p=self.aplica_temperature(strategie,
                                                                                                  env.nrMutari)))

        if can_stop and self.conf_joc.resign_threshold is not None and \
                valoarePozitie <= self.conf_joc.resign_threshold \
                and env.nrMutari > self.conf_joc.min_resign_turn:
            # noinspection PyTypeChecker
            return None if not self.for_web else (None, fens, root_values)
        else:
            #self.mutari.append([list(strategie)])
            self.mutari.append([list(strategie)])  # cu istorie
            #self.mutari.append([env.vezi_fen, list(strategie)]) # fara istorie
            return self.config.etichete[strategieFinala] if not self.for_web else (self.config.etichete[strategieFinala],
                                                                             fens, root_values, self.arbore)

    def cauta_toate_mutarile(self, env):
        """
            Caut in mod paralel mutari si o intorc pe cea cu valoare maxima
        """
        futures = []
        with ThreadPoolExecutor(max_workers=self.conf_joc.search_threads) as executor:
            for i in range(self.conf_joc.simulation_num_per_move):
                futures.append(executor.submit(self.cauta_mutari, env=env.copiaza(), is_root_node=True))
        vals = [f.result() for f in futures]
        if self.for_web:
            return sorted(vals, key=lambda x: float(-x[0]))
        else:
            return np.max(vals), vals[0]

    def cauta_mutari(self, env: Env, is_root_node=False, adancime=0, parent=None) -> float:
        """
        prob_medie, valoare_frunza sunt valorile pentru jucatorul curent(mereu alb)
        prob_retea - strategia prezisa de retea pentru urmatorul jucator
        Functia cauta mutari noi, le adauga in arbore si returneaza cea mai buna mutare din pozitia curenta.
        """
        if env.sfarsit:
            if env.castigator == Winner.draw:
                return 0 if not self.for_web else (0, env.tabla.fen())
            # assert env.whitewon != env.muta_alb    # side to move can't be castigator!
            return -1 if not self.for_web else (-1, env.tabla.fen())
        state = codificaPozitia(env)
        with self.node_lock[state]:
            if state not in self.arbore:    # daca gasesc un nod care nu e in arbore, il adaug
                strategie_frunza, valoare_frunza = self.evalueaza_conform_retelei(env.copiaza())
                self.arbore[state].prob_retea = strategie_frunza
                #self.arbore[state].parent = parent
                return valoare_frunza if not self.for_web else (valoare_frunza, env.tabla.fen())

            # Partea de selectare din MCTS / daca nodul e in arbore
            mutare = self.selecteaza_pentru_explorare(env.copiaza(), is_root_node)

            virtual_loss = self.conf_joc.virtual_loss

            nodArbore = self.arbore[state]
            detaliiNodArbore = nodArbore.detalii[mutare]

            nodArbore.sum_n += virtual_loss
            detaliiNodArbore.nr_viz += virtual_loss
            detaliiNodArbore.cumul += -virtual_loss
            detaliiNodArbore.prob_medie = detaliiNodArbore.cumul / detaliiNodArbore.nr_viz
        env.muta(mutare.uci())
        # if self.for_web:
        #     print('\nsearch\n', env.tabla, '\n')

        if not self.for_web:
            valoare_frunza = self.cauta_mutari(env.copiaza(), adancime=adancime + 1, parent=codificaPozitia(env))  # next move from enemy POV
        else:
            valoare_frunza = self.cauta_mutari(env.copiaza(), adancime=adancime + 1, parent=codificaPozitia(env))[0]

        valoare_frunza = -valoare_frunza

        # Partea de backup din MCTS
        # on returning search path
        # copiazaDeLa: N, W, Q
        with self.node_lock[state]:
            nodArbore.sum_n += -virtual_loss + 1
            nodArbore.fen = env.tabla.fen()

            detaliiNodArbore.nr_viz += -virtual_loss + 1
            detaliiNodArbore.cumul += virtual_loss + valoare_frunza
            detaliiNodArbore.prob_medie = detaliiNodArbore.cumul / detaliiNodArbore.nr_viz
            #nodArbore.adancime = adancime
            #nodArbore.children.append(codificaPozitia(env))
            #print(nodArbore.adancime, nodArbore.children)

        return valoare_frunza if not self.for_web else (valoare_frunza, env.tabla.fen())

    def evalueaza_conform_retelei(self, env) -> (np.ndarray, float):
        """ La fiecare pozitie nou intalnita, apelez aceasta functie pentru a adauga detaliile in nod.
                    APELEZ DOAR CU LOCK !!!
        """
        state_planes = env.reprezentareTabla()
        #print('\npredict\n', chess.Board(intoarceTabla(env.tabla.fen(), env.muta_alb)), '\n')
        #print(state_planes[0])
        strategie_frunza, valoare_frunza = self.predict(state_planes)
        if not env.muta_alb:
            strategie_frunza = Config.inv_strategie(strategie_frunza)
        return strategie_frunza, valoare_frunza

    def predict(self, state_planes):
        pipe = self.pipe_pool.pop()
        pipe.send(state_planes)
        ret = pipe.recv()
        self.pipe_pool.append(pipe)
        return ret

    def selecteaza_pentru_explorare(self, env, radacina, ) -> chess.Move:
        """
        Cauta urmatorul nod pentru explorare. Alege nodul in functie de mutarea care ar maximiza valoarea pozitiei
         (prob_medie)
        :param Environment env: env to look for the next mutari within
        :param radacina: whether this is for the root node of the MCTS search.
        :return chess.Move: the move to explore
        """

        state = codificaPozitia(env)

        nodArbore = self.arbore[state]

        if nodArbore.prob_retea is not None:
            tot_p = 1e-8
            for mov in env.tabla.legal_moves:
                mov_p = nodArbore.prob_retea[self.san_catre_nr[mov]]
                nodArbore.detalii[mov].prob_retea = mov_p
                tot_p += mov_p
            for detaliiNod in nodArbore.detalii.values():
                detaliiNod.prob_retea /= tot_p
            nodArbore.prob_retea = None

        xx_ = np.sqrt(nodArbore.sum_n + 1)  # sqrt of sum(N(s, b); for all b)
        e = self.conf_joc.noise_eps
        c_puct = self.conf_joc.c_puct
        dir_alpha = self.conf_joc.dirichlet_alpha

        valoareMax = -999
        mutareMax = None
        if radacina:
            noise = np.random.dirichlet([dir_alpha] * len(nodArbore.detalii))

        i = 0
        for mutare, detaliiNod in nodArbore.detalii.items():
            p_ = detaliiNod.prob_retea
            if radacina:
                p_ = (1 - e) * p_ + e * noise[i]
                i += 1
            valCurenta = detaliiNod.prob_medie + c_puct * p_ * xx_ / (1 + detaliiNod.nr_viz)
            if valCurenta > valoareMax:
                valoareMax = valCurenta
                mutareMax = mutare
        return mutareMax

    def aplica_temperature(self, policy, turn):
        """
        Aplica zgomot in strategia de mutare, in functie de cat de mult s-a jucat.
        Vreau ca primele mutari sa fie random atunci cand joaca singur, apoi din ce in ce mai precise, pentru a
        avea o baza de date cat mai diversa.
        """
        tau = np.power(self.conf_joc.tau_decay_rate, turn + 1)
        if tau < 0.1:
            tau = 0
        if tau == 0:
            action = np.argmax(policy)
            #print(policy, policy[gasesteMutare])
            ret = np.zeros(self.nr_etichete)
            ret[action] = 1.0
            return ret
        else:
            ret = np.power(policy, 1 / tau)
            ret /= np.sum(ret)
            return ret

    def calc_strategie(self, env):
        state = codificaPozitia(env)
        nodArbore = self.arbore[state]
        strategie = np.zeros(self.nr_etichete)
        for mutare, detalii in nodArbore.detalii.items():
            strategie[self.san_catre_nr[mutare]] = detalii.nr_viz
        #print(policy[policy != 0])
        strategie /= np.sum(strategie)
        #print(policy[policy != 0])

        return strategie

    def mutare_supervizat(self, fen, mutare, pondere=1):
        """
        Functie folosita pentru invatare supervizata. Transforma mutarile din baza de date de la FICS in planuri pe care
        le poate intelege reteaua. Ponderez rezultatele in functie de cat de bun este jucatorul din baza de date.
        """
        strategie = np.zeros(self.nr_etichete)

        k = self.san_catre_nr[chess.Move.from_uci(mutare)]
        strategie[k] = pondere

        self.mutari.append([list(strategie)]) # cu istorie
        #self.mutari.append([fen, list(strategie)]) # fara istorie
        return mutare

    def adauga_castigator(self, castigator):
        """
        Adauga castigatorul in baza de date, pentru a putea invata valoarea unui joc. ( 0, 1, 0.5 )
        """
        for mutare in self.mutari:
            mutare += [castigator]

def codificaPozitia(env: Env) -> str:
    return env.tabla.fen()

# def codificaPozitia(env: Env) -> str:
#     fen = env.tabla.fen().rsplit(' ', 1)  # scot numarul mutarii
#     return fen[0]

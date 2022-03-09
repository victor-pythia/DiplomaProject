import numpy as np
import chess, chess.variant
from enum import Enum
import copy
from _collections import deque
from random import randint, choice

pieces_order = 'KQRBNPkqrbnp'  # 12x8x8
castling_order = 'KQkq'  # 4x8x8
ind = {pieces_order[i]: i for i in range(12)}

Winner = Enum("Winner", "black white draw")


class Env:
    def __init__(self):
        self.tabla = chess.variant.AntichessBoard()
        fen = self.tabla.fen()
        self.stivaFen = [fen]
        self.stivaPlanuri = [np.asarray([np.full((8, 8), 0) for _ in range(12)]) for _ in range(3)]
        self.stivaPlanuri.append(fen_catre_planuri(intoarceTabla(fen, muta_negru(fen))))
        self.castigator = None
        self.rezultat = None
        self.nrMutari = 0
        self.cedat = False

    def reseteaza(self, pozitie_aleatoare=False, nr_aleator=20):
        self.tabla = chess.variant.AntichessBoard()
        self.castigator = None
        self.cedat = False
        self.nrMutari = 0
        fen = self.tabla.fen()
        self.stivaPlanuri = [fen]
        self.stivaPlanuri = [np.asarray([np.full((8, 8), 0) for _ in range(12)]) for _ in range(3)]
        self.stivaPlanuri.append(fen_catre_planuri(intoarceTabla(fen, muta_negru(fen))))
        if pozitie_aleatoare:
            nr_mutari = randint(0, nr_aleator)
            for _ in range(nr_mutari):
                mutare = choice([self.tabla.san(m) for m in self.tabla.legal_moves])
                self.muta(mutare, san=True)
        return self

    def copiazaDeLa(self, tabla):
        self.tabla = chess.variant.AntichessBoard()
        self.stivaPlanuri = tabla.stivaPlanuri.copiaza()
        self.stivaFen = tabla.stivaFen.copiaza()
        self.castigator = None
        self.cedat = False
        return self

    def reprezentareTabla(self):
        return planuriAZ(self.tabla.fen(), self.stivaPlanuri)

    def muta(self, mutare, check_over=True, san=False):
        if check_over and mutare is None:
            self.cedeaza()
            return
        if san:
            self.tabla.push_san(mutare)
        else:
            self.tabla.push_uci(mutare)
        self.nrMutari += 1
        if check_over and self.tabla.result(claim_draw=True) != "*":
            self.joc_incheiat()

        fen = self.tabla.fen()
        self.stivaPlanuri.append(fen_catre_planuri(intoarceTabla(fen, muta_negru(fen))))
        self.stivaFen.append(fen)
        del self.stivaPlanuri[0]

    def cedeaza(self):
        self.cedat = True
        if self.muta_alb:  # WHITE RESIGNED!
            self.castigator = Winner.black
            self.rezultat = "0-1"
        else:
            self.castigator = Winner.white
            self.rezultat = "1-0"

    def copiaza(self):
        env = copy.copy(self)
        env.tabla = copy.copy(self.tabla)
        env.stivaPlanuri = self.stivaPlanuri.copy()
        env.stivaFen = self.stivaFen.copy()
        return env

    def joc_incheiat(self):
        if self.castigator is None:
            self.rezultat = self.tabla.result(claim_draw=True)
            if self.rezultat == '1-0':
                self.castigator = Winner.white
            elif self.rezultat == '0-1':
                self.castigator = Winner.black
            else:
                self.castigator = Winner.draw

    def analizaObiectiva(self):
        score = evaluare_obiectiva(self.tabla.fen(), absolute=True)
        if abs(score) < 0.01:
            self.castigator = Winner.draw
            self.rezultat = "1/2-1/2"
        elif score > 0:
            self.castigator = Winner.white
            self.rezultat = "1-0"
        else:
            self.castigator = Winner.black
            self.rezultat = "0-1"

    @property
    def sfarsit(self):
        return self.castigator is not None

    @property
    def a_castigat_alb(self):
        return self.castigator == Winner.white

    @property
    def muta_alb(self):
        return self.tabla.turn == chess.WHITE

    @property
    def vezi_fen(self):
        return self.tabla.fen()


def planuriAZ(fen, movestack):
    fen = intoarceTabla(fen, muta_negru(fen))
    return reprezenetari_plus_detalii(fen, movestack)


def reprezenetari_plus_detalii(fen , stivaMutari):
    plan_detalii = planuri_detaliu(fen)
    ret = np.vstack((plan_detalii, np.vstack(stivaMutari)))  # 7 + 4*12
    #plan_tabla = fen_catre_planuri(fen)
    #ret = np.vstack((plan_detalii, plan_tabla))
    assert ret.shape == (55, 8, 8)
    return ret


def intoarceTabla(fen, flip=False):
    if not flip:
        return fen
    foo = fen.split(' ')
    rows = foo[0].split('/')

    def mici_mari(a):
        if a.isalpha():
            return a.lower() if a.isupper() else a.upper()
        return a

    def schimba_litere(linie):
        return "".join([mici_mari(litera) for litera in linie])

    return "/".join([schimba_litere(row) for row in reversed(rows)]) \
           + " " + ('w' if foo[1] == 'b' else 'b') \
           + " " + "".join(sorted(schimba_litere(foo[2]))) \
           + " " + foo[3] + " " + foo[4] + " " + foo[5]


def planuri_detaliu(fen):
    blocks = fen.split(' ')
    en_passant = np.zeros((8, 8), dtype=np.float32)
    if blocks[3] != '-':
        eps = algebric_in_coordonate(blocks[3])
        en_passant[eps[0]][eps[1]] = 1

    fifty_move_count = int(blocks[4])
    fifty_move = np.full((8, 8), fifty_move_count, dtype=np.float32)

    castling = blocks[2]
    detail_planes = [
        np.full((8, 8), int(blocks[1] == 'w'), dtype=np.float32),
        np.full((8, 8), int('K' in castling), dtype=np.float32),
        np.full((8, 8), int('Q' in castling), dtype=np.float32),
        np.full((8, 8), int('k' in castling), dtype=np.float32),
        np.full((8, 8), int('prob_medie' in castling), dtype=np.float32),
        fifty_move,
        en_passant]

    ret = np.asarray(detail_planes, dtype=np.float32)
    assert ret.shape == (7, 8, 8)
    return ret


def evaluare_obiectiva(fen, absolute=False):
    piece_vals = {'K': 3, 'Q': 14, 'R': 5, 'B': 3.25, 'N': 3, 'P': 1}
    ans = 0.0
    tot = 0
    for c in fen.split(' ')[0]:
        if not c.isalpha():
            continue

        if c.isupper():
            ans -= piece_vals[c]
            tot -= piece_vals[c]
        else:
            ans += piece_vals[c.upper()]
            tot -= piece_vals[c.upper()]
    v = ans / tot
    if not absolute and muta_negru(fen):
        v = -v
    assert abs(v) < 1
    return np.tanh(v * 3)


def algebric_in_coordonate(alg):
    linie = 8 - int(alg[1])  # 0-7
    coloana = ord(alg[0]) - ord('a')  # 0-7
    return linie, coloana


def coordonata_in_algebric(coord):
    letter = chr(ord('a') + coord[1])
    number = str(8 - coord[0])
    return letter + number


def fen_catre_planuri(fen):
    tabla = adauga_unu(fen)
    planuri = np.zeros(shape=(12, 8, 8), dtype=np.float32)

    for linie in range(8):
        for coloana in range(8):
            v = tabla[linie * 8 + coloana]
            if v.isalpha():
                planuri[ind[v]][linie][coloana] = 1
    assert planuri.shape == (12, 8, 8)
    return planuri


def adauga_unu(board_san):
    board_san = board_san.split(" ")[0]
    board_san = board_san.replace("2", "11")
    board_san = board_san.replace("3", "111")
    board_san = board_san.replace("4", "1111")
    board_san = board_san.replace("5", "11111")
    board_san = board_san.replace("6", "111111")
    board_san = board_san.replace("7", "1111111")
    board_san = board_san.replace("8", "11111111")
    return board_san.replace("/", "")


def muta_negru(fen):
    return fen.split(" ")[1] == 'b'

import os
import numpy as np


class ConfiguratieJucatorUman:
    def __init__(self):
        self.numar_simulari_pe_mutare = 1600
        self.m_fire_executie = 2
        self.conf_puct = 1
        self.bruiaj = 0
        self.rata_micsorare_tau = 0
        self.threshold_cedare = None

    def modifica_configuratie(self, pc):
        pc.simulation_num_per_move = self.numar_simulari_pe_mutare
        pc.search_threads *= self.m_fire_executie
        pc.conf_puct = self.conf_puct
        pc.noise_eps = self.bruiaj
        pc.tau_decay_rate = self.rata_micsorare_tau
        pc.threshold_cedare = self.threshold_cedare
        pc.max_game_length = 999999


class Options:
    new = False


class ConfigProiect:
    def __init__(self):
        self.ramdisk = False
        self.folder_proiect = "."
        #self.folder_date = "/media/lilmarco/Seagate_HDD/LICENTA_DATE"
        self.folder_date = "./data"
        self.folder_model = "./model"
        if self.ramdisk:
            self.cale_config_cel_mai_bun_model = os.path.join('/tmp/ramdisk', "model_best_config.json")
            self.cale_parametrii_cel_mai_bun_model = os.path.join('/tmp/ramdisk', "model_best_weight.h5")
        else:
            self.cale_config_cel_mai_bun_model = os.path.join(self.folder_model, "model_best_config.json")
            self.cale_parametrii_cel_mai_bun_model = os.path.join(self.folder_model, "model_best_weight.h5")

        self.fld_date_pattern = "supervizat_%s.json"
        self.fld_date = os.path.join(self.folder_date, "Experienta")

        self.fld_date_spvz_pattern = "supervizat_%s.json"
        self.fld_date_spvz = os.path.join(self.folder_date, "Supervizat")

        if self.ramdisk:
            self.fld_date_spvz_hdd = '/media/lilmarco/Seagate_HDD/LICENTA_DATE/THE_DB'
            self.fld_spvz_train_data = '/tmp/ramdisk/Train'
        else:
            self.fld_date_spvz_hdd = '/media/lilmarco/Seagate_HDD/LICENTA_DATE/THE_DB'
            self.fld_spvz_train_data = '/media/lilmarco/Seagate_HDD/LICENTA_DATE/Train'
        self.fld_date_spvz_done = '/media/lilmarco/Seagate_HDD/LICENTA_DATE/Done'

        if self.ramdisk:
            self.folder_candidati = os.path.join('/tmp/ramdisk', "Candidati")
        else:
            self.folder_candidati = os.path.join(self.folder_model, "Candidati")
        self.template_nume_candidati = "candidat_%s"
        self.candidat_config = "candidat_config.json"
        self.candidat_parametrii = "candidat_weight.h5"

        self.fld_log = os.path.join(self.folder_proiect, "logs")
        self.fld_main_log = os.path.join(self.fld_log, "main.log")

    def creeaza_fld(self):
        fld = [self.folder_proiect, self.folder_date, self.folder_model, self.fld_date, self.fld_log,
               self.folder_candidati]
        for d in fld:
            if not os.path.exists(d):
                os.makedirs(d)


def etichete_inversate():
    # Transforma etichetele in format UCI.
    def inloc(x):
        return "".join([(str(9 - int(a)) if a.isdigit() else a) for a in x])

    return [inloc(x) for x in creeaza_etichete()]


def creeaza_etichete():
    # Returneaza o lista de etichete transformate in UCI
    etichete_arr = []
    litere = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numere = ['1', '2', '3', '4', '5', '6', '7', '8']
    promovari = ['q', 'r', 'b', 'n', 'k']

    for l1 in range(8):
        for n1 in range(8):
            destinatii = [(t, n1) for t in range(8)] + \
                           [(l1, t) for t in range(8)] + \
                           [(l1 + t, n1 + t) for t in range(-7, 8)] + \
                           [(l1 + t, n1 - t) for t in range(-7, 8)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
            for (l2, n2) in destinatii:
                if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):
                    move = litere[l1] + numere[n1] + litere[l2] + numere[n2]
                    etichete_arr.append(move)
    for l1 in range(8):
        l = litere[l1]
        for p in promovari:
            etichete_arr.append(l + '2' + l + '1' + p)
            etichete_arr.append(l + '7' + l + '8' + p)
            if l1 > 0:
                l_l = litere[l1 - 1]
                etichete_arr.append(l + '2' + l_l + '1' + p)
                etichete_arr.append(l + '7' + l_l + '8' + p)
            if l1 < 7:
                l_r = litere[l1 + 1]
                etichete_arr.append(l + '2' + l_r + '1' + p)
                etichete_arr.append(l + '7' + l_r + '8' + p)
    return etichete_arr


class Config:
    etichete = creeaza_etichete()
    nr_etichete = int(len(etichete))
    inv_etichete = etichete_inversate()
    idx_neinversat = None

    def __init__(self, config_type="mini"):
        self.opts = Options()
        self.confProiect = ConfigProiect()

        if config_type == "mini":
            import configs.mini as c
        elif config_type == "normal":
            import configs.normal as c
        else:
            raise RuntimeError(f"unknown config_type: {config_type}")
        self.model = c.ConfiguratieModel()
        self.conf_joc = c.ConfigJoc()
        self.conf_joc_spvz = c.ConfigSupervizat()
        self.conf_antr = c.ConfigAntrenament()
        self.eval = c.ConvigEvaluare()
        self.etichete = Config.etichete
        self.nr_etichete = Config.nr_etichete
        self.inv_etichete = Config.inv_etichete

    @staticmethod
    def inv_strategie(pol):
        # Transforma strategia din alb in negru si invers
        return np.asarray([pol[ind] for ind in Config.idx_neinversat])


Config.idx_neinversat = [Config.etichete.index(x) for x in Config.inv_etichete]


def _fld_proiect():
    d = os.path.dirname
    return d(d(d(os.path.abspath(__file__))))


def _fld_date():
    return os.path.join(_fld_proiect(), "data")

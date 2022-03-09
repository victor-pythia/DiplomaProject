from Config import Config
from multiprocessing import set_start_method
import os
import sys
import argparse
from lib.logger import setup_logger

_PATH_ = os.path.dirname(os.path.dirname(__file__))

CMD_LIST = ['self', 'opt', 'eval', 'sl', 'play']
LISTA_JOCURI = ["chess", "antichess"]


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", help="what to do", choices=CMD_LIST)
    parser.add_argument("--new", help="run from new best model", action="store_true")
    parser.add_argument("--type", help="use normal setting", default="mini")
    parser.add_argument("--total", help="set TrainerConfig.start_total_steps", type=int)
    parser.add_argument("--epoch", help="Numarul de epoci", type=int, default=1)
    parser.add_argument("--tip", help="Tipul de joc", type=str, default="chess", choices=LISTA_JOCURI)
    parser.add_argument("--interval", help="Intervalul indecsilor de joc din fisierul pgn", type=str, default='0 1000000')
    parser.add_argument("--date", help='Supervizat sau rfl', choices=['sup', 'rfl'], default='rfl')
    return parser


if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    Config = Config()
    Config.conf_antr.epoch_to_checkpoint = args.epoch
    Config.conf_joc_spvz.interval = tuple(map(int, args.interval.split()))
    if args.date == 'sup':
        Config.confProiect.fld_date = Config.confProiect.fld_date_spvz
        Config.confProiect.fld_date_pattern = Config.confProiect.fld_date_spvz_pattern
    setup_logger(Config.confProiect.fld_main_log)
    set_start_method('spawn')
    sys.setrecursionlimit(10000000)
    if args.cmd == 'self':
        from SelfPlay import start
        start(Config)

    elif args.cmd == 'opt':
        from Antreneaza import start
        start(Config)

    elif args.cmd == 'eval':
        from Evalueaza import start
        start(Config)

    elif args.cmd == 'sl':
        from Supervizat import start
        start(Config)

    elif args.cmd == 'play':
        from WebServer.ServerJoc import start
        from Config import ConfiguratieJucatorUman
        ConfiguratieJucatorUman().modifica_configuratie(Config.conf_joc)
        start(Config)

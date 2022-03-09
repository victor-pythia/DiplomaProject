class ConvigEvaluare:
    def __init__(self):
        self.vram_frac = 1.0
        self.game_num = 50
        self.replace_rate = 0.55
        self.conf_joc = ConfigJoc()
        self.conf_joc.simulation_num_per_move = 600
        self.conf_joc.c_puct = 1    # mic - joc cum stiu, mare - explorez
        self.conf_joc.tau_decay_rate = 0.5
        self.conf_joc.noise_eps = 0
        self.evaluate_latest_first = True
        self.max_game_length = 250


class ConfigSupervizat:
    def __init__(self):
        self.min_elo_policy = 1400
        self.max_elo_policy = 1800
        self.sl_nb_game_in_file = 100
        self.nb_game_in_file = 50
        self.max_file_num = 1000
        self.interval = (0, 1000000)


class ConfigJoc:
    def __init__(self):
        self.nivelStockfish = 1
        self.max_processes = 28
        self.search_threads = 11
        self.vram_frac = 1.0
        self.simulation_num_per_move = 200
        self.logging_thinking = False
        self.c_puct = 4
        self.noise_eps = 0.3
        self.dirichlet_alpha = 0.3
        self.tau_decay_rate = 0.7
        self.virtual_loss = 3
        self.resign_threshold = -0.8
        self.min_resign_turn = 50
        self.max_game_length = 200


class ConfigAntrenament:
    def __init__(self):
        self.min_data_size_to_learn = 0
        self.cleaning_processes = 20
        self.vram_frac = 0.95
        self.batch_size = 300
        self.epoch_to_checkpoint = 1
        self.dataset_size = 1000000
        self.start_total_steps = 0
        self.save_model_steps = 75
        self.load_data_steps = 100
        self.loss_weights = [1.0, 1.5]     # [strategie, valoare]


class ConfiguratieModel:
    nr_filtre = 256
    dim_primul_kernel = 5
    dim_kernel = 3
    nr_bloc_rezid = 20
    l2_reg = 1e-4
    nr_fc_valoare = 256
    dim_input = 55
    print_summary = True

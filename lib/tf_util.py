def set_session_config(per_process_gpu_memory_fraction=None, allow_growth=None):
    from tensorflow import ConfigProto, Session, get_default_graph
    from keras.backend import set_session
    from time import sleep

    result = False
    while not result:
        try:
            conf = ConfigProto(allow_soft_placement=True, device_count={'CPU': 1, 'GPU': 1}, allow_growth=allow_growth)
            conf.gpu_options.allow_growth = True
            conf.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
            sess = Session(config=conf)
            graph = get_default_graph()
            result = True
        except:
            sleep(1)
    set_session(sess)
    return sess, graph

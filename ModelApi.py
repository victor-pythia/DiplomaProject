from multiprocessing import connection, Pipe
from threading import Thread
from keras.backend import set_session

import numpy as np

from Config import Config


class ModelApi:
    def __init__(self, agent):
        self.agent = agent
        self.pipes = []

    def start(self):
        worker = Thread(target=self._predict_batch_worker, name="worker")
        worker.daemon = True
        worker.start()

    def create_pipe(self):
        api, thread = Pipe()
        self.pipes.append(api)
        return thread

    def _predict_batch_worker(self):
        while True:
            ready = connection.wait(self.pipes,  timeout=20)
            if not ready:
                continue
            data, result_pipes = [], []
            for pipe in ready:
                while pipe.poll():
                    data.append(pipe.recv())
                    result_pipes.append(pipe)

            data = np.asarray(data, dtype=np.float32)
            with self.agent.session.graph.as_default():
                set_session(self.agent.session)
                strategie, valori = self.agent.model.predict_on_batch(data)

            for pipe, p, v in zip(result_pipes, strategie, valori):
                pipe.send((p, float(v)))

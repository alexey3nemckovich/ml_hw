from gensim.models.callbacks import CallbackAny2Vec


class ProgressCallback(CallbackAny2Vec):
    def __init__(self, logger):
        self.last_logged_progress = 0
        self.logger = logger
        self.current_epoch = 0

    def on_epoch_end(self, model):
        self.current_epoch += 1

        progress = int((self.current_epoch/model.epochs) * 100)
        if progress > self.last_logged_progress and progress > 0:
            self.logger.info(f"Training progress: {progress}%")
            self.last_logged_progress = progress

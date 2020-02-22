import tensorflow as tf
import os
import datetime
import shutil

class Logger:
    def __init__(self, ticker, episodes):

        self.logdir   = os.path.join('logs', f'run_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{ticker}')
        self.episodes = episodes

        self._delete_runs()
        
        self.writers = self._create_writers()

    def _create_writers(self):
        """
        Create different writers for different runs
        """
        logdirs = [os.path.join(self.logdir, f'run_{i}') for i in range(self.episodes)]
        return [tf.summary.create_file_writer(logdir) for logdir in logdirs]

    def _delete_runs(self):
        """
        Delete all old runs
        """
        filelist = [ f for f in os.listdir('logs') if f.startswith('run_') ]

        for f in filelist:
            shutil.rmtree(os.path.join('logs', f))

    def log_scalar(self, episode, name, value, step):
        """
        Log a scalar variable
        """

        writer = self.writers[episode]

        with writer.as_default():
            tf.summary.scalar(name, value, step)
            writer.flush()
import tensorflow as tf
import os
import datetime

class Logger:
    def __init__(self, name):
        logdir = os.path.join('data', 'logs', name)
        self.writer = tf.summary.create_file_writer(logdir)

    def log_scalar(self, name, value, step):
        """
        Log a scalar variable
        """
        with self.writer.as_default():
            tf.summary.scalar(name, value, step)
            self.writer.flush()
import logging

from abc import ABC


class Logging(ABC):
    def __init__(self):
        super(Logging, self).__init__()
        self.logger = logging.getLogger(f'nnqs.{self.__class__.__name__}')

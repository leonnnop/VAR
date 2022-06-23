from .mainmodel import Reasoner

def get_model(config):
    return Reasoner(config)
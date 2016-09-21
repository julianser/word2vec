import theano
from theano import tensor as T

def main():
    data = load_data()
    data_train, data_eval, data_test = data


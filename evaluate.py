from assign import vars
from assign.scripts.evaluate_model import evaluate

from argparse import ArgumentParser


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('model', help='model to evaluate')
	parser.add_argument('weights', help='path to weights file')

	args = parser.parse_args()

	evaluate(vars, args.model+'_model', args.weights)
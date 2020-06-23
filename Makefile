

all: train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz


%.gz:
	curl -O -J http://yann.lecun.com/exdb/mnist/$@



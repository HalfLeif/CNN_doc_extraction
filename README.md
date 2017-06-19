Master Thesis by Leif Schelin, June 2017, at Chalmers University of Technology. [Source](https://github.com/HalfLeif/CNN_doc_extraction).

We attempt to classify year of Swedish civil population records by deep learning using CNNs and attention models. The labels have been extracted from GedcomX using [GedcomXExtractor](https://github.com/HalfLeif/GedcomXExtract) into csv-files.

There are three runnable modules described below.

The corresponding Master Thesis report can be found [here](https://github.com/HalfLeif/Master-Thesis---Waypointing).


# Requirements

Python 3.6 or later.

Pip dependencies:
- `numpy`
- `tensorflow`
- `python-gflags`


# Module main

Loads a network from file or creates a newly initialized network.
The list `lambdas` contains all steps that should be computed. The operations are added to Tensorflow's computation graph and they are evaluated in order in the same session. Additionally can time each operation.

Supported operations:
- Load model from file: `ops.model_io.lazyLoadModel`. If no model was loaded, the parameters are initialized randomly.
- Train model: `ops.train.lazyTrain`.
- Test model: `ops.test.lazyTest`.
- Run model on dataset and output logits to file: `ops.classify.lazyClassify`. Logits can be used further by module `page_sequence`.
- Start timer and output spent cputime: `ops.time.lazyStart` and `ops.time.lazyClock`.

Requires that the data directories are properly assigned using flags, see `loading.load_swe`.

The number of threads can be adjusted using flag `NUM_THREADS`.


# Module countlabels

Counts how many times each label occurs in a dataset.

Requires that the data directories are properly assigned using flags, see `loading.load_swe`.


# Module page_sequence

Loads classifications (logits) of a dataset. Prints the dataset as an ordered sequence with the predicted label next to the true label.

Optionally performs post-processing `optimizeBooks` and also prints the post-processed prediction.

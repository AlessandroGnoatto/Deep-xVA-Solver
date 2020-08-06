# [Deep xVA Solver](https://arxiv.org/abs/2005.02633) in TensorFlow (2.0)


## How to run the examples

Currently we offer two examples from the paper. We compute the exposure of:

* A forward on a single underlying.
* A basket call option on 100 underlyings.
* Recursive Computation of FVA on a forward
* Recursive Computation of BCVA on a basket call with 100 underlyings.

To run the examples, simply rung the associated scripts

* forward.py
* basketCall.py
* fvaForward.py
* basketCallWithCVA.py


## Aknowledgements.
We are grateful to Chang Jiang for the help in the conversion of our code base from Tensorflow 1.x to Tensorflow 2.2.


## Dependencies

* [TensorFlow >=2.0](https://www.tensorflow.org/)


## Reference
[1] Gnoatto, A., Picarelli, A., and Reisinger, C. Deep xVA solver - A neural network based counterparty credit risk management framework. [[arXiv]](https://arxiv.org/abs/2005.02633)

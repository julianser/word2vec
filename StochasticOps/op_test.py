# Make sure this code runs with:
# THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float64 python op_test.py
# THEANO_FLAGS=mode=FAST_COMPILE,device=cpu,floatX=float64 python op_test.py
import theano
import theano.tensor as T

import numpy

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
trng = RandomStreams(12345)

from theano import pp

from utils import *


rng = numpy.random.RandomState(1234)

x = T.vector()
pvals = SoftMax(x)

#y = trng.multinomial(pvals=pvals, dtype='int32')
#y = trng.multinomial_meanfield(pvals=pvals, dtype='int32')
y = trng.multinomial_straightthrough(pvals=pvals, dtype='int64')

emb = theano.shared(value=NormalInit(rng, 5, 1))
y_projection = T.dot(y, emb)

# Print un-optimized Theano graph
#print 'theano.printing.debugprint(y_projection)', theano.printing.debugprint(y_projection)

f = theano.function([x], [y_projection, pvals, y, T.grad(T.sum(y_projection), pvals)])

#f = theano.function([x], [y_projection, pvals, y])

# Print optimized Theano graph
#print 'theano.printing.debugprint(f.maker.fgraph.outputs[0])', theano.printing.debugprint(f.maker.fgraph.outputs[0])

inp = rng.rand(5).astype('float32')
out, out_x, out_y, out_grad = f(inp)

print 'out', out, out_x, out_y, out_grad

#f_grad = theano.function([x], T.grad(T.sum(y_projection), pvals))
#out_grad = f_grad(inp)
#print 'out_grad', out_grad


### Tests

do_tests = True

if do_tests:
    fail_count = 0
    for i in range(100):
        print 'test i', i
        inp = rng.rand(5).astype('float32')
        out, out_x, out_y, out_grad = f(inp)

        #print 'out', out, out_x, out_y

        #f_grad = theano.function([x], T.grad(T.sum(y_projection), pvals))
        #out_grad = f_grad(inp)
        #print 'out_grad', out_grad.shape, out_grad
        
        for k in range(5):
            if numpy.abs(out_grad[0, k]) > 0.000001:
                if not numpy.abs(out_y[0, k]) > 0.000001:
                    fail_count += 1
                    print 'FAILED!'
                    print 'out', out, out_x, out_y
                    print 'out_grad', out_grad.shape, out_grad


    print 'fail_count', fail_count


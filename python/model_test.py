"""
Tests for model saving and loading, including for user-defined models.
"""
from __future__ import print_function
import dynet as dy
import numpy
import os

# first, define three user-defined classes

class Transfer(Saveable):
    def __init__(self, nin, nout, act, model):
        self.act = act
        self.W = model.add_parameters((nout, nin))
        self.b = model.add_parameters(nout)
        self.nin = nin
        self.nout = nout

    def __call__(self, x):
        W,b=map(dy.parameter, [self.W, self.b])
        return self.act(W*x+b)

    def get_components(self):
        return [self.W, self.b]

    def restore_components(self, components):
        self.W, self.b = components

class MultiTransfer(Saveable):
    def __init__(self, sizes, act, model):
        self.transfers = []
        for nin,nout in zip(sizes,sizes[1:]):
            self.transfers.append(Transfer(nin,nout,act,model))

    def __call__(self, x):
        for t in self.transfers:
            x = t(x)
        return x

    def get_components(self):
        return self.transfers

    def restore_components(self, components):
        self.transfers = components

class NoParameters(Saveable):
    def __init__(self, act):
        self.act = act
    def __call__(self, in_expr):
        return self.act(dy.cmult(in_expr))
    def get_components(self): return []
    def restore_components(self,components):pass

def old_style_save_and_load():
    # create a model and add parameters.
    m = dy.Model()
    a = m.add_parameters((100,100))
    b = m.add_lookup_parameters((20,2))
    t1 = Transfer(5,6,dy.softmax, m)
    t2 = Transfer(7,8,dy.softmax, m)
    tt = MultiTransfer([10,10,10,10],dy.tanh, m)
    c = m.add_parameters((100))
    lb = dy.LSTMBuilder(1,2,3,m)
    lb2 = dy.LSTMBuilder(2,4,4,m)
    # save
    m.save("test1")

    # create new model (same parameters):
    m2 = dy.Model()
    a2 = m2.add_parameters((100,100))
    b2 = m2.add_lookup_parameters((20,2))
    t12 = Transfer(5,6,dy.softmax, m2)
    t22 = Transfer(7,8,dy.softmax, m2)
    tt2 = MultiTransfer([10,10,10,10],dy.tanh, m2)
    c2 = m2.add_parameters((100))
    lb2 = dy.LSTMBuilder(1,2,3,m2)
    lb22 = dy.LSTMBuilder(2,4,4,m2)

    # parameters should be different
    for p1,p2 in [(a,a2),(b,b2),(c,c2),(t1.W,t12.W),(tt.transfers[0].W,tt2.transfers[0].W)]:
        assert(not numpy.array_equal(p1.as_array(), p2.as_array()))

    m2.load("test1")

    # parameters should be same
    for p1,p2 in [(a,a2),(b,b2),(c,c2),(t1.W,t12.W),(tt.transfers[0].W,tt2.transfers[0].W)]:
        assert(numpy.array_equal(p1.as_array(), p2.as_array()))
    

    os.remove("test1")

old_style_save_and_load()

def new_style_save_and_load():
    # create a model and add parameters.
    m = dy.Model()
    a = m.add_parameters((100,100))
    b = m.add_lookup_parameters((20,2))
    t1 = Transfer(5,6,dy.softmax, m)
    t2 = Transfer(7,8,dy.softmax, m)
    tt = MultiTransfer([10,10,10,10],dy.tanh, m)
    c = m.add_parameters((100))
    lb = dy.LSTMBuilder(1,2,3,m)
    lb2 = dy.LSTMBuilder(2,4,4,m)
    np = NoParameters(dy.tanh)
    # save
    m.save("test_new",[a,b,t1,t2,tt,c,lb,lb2,np])
    m.save("test_new_r",[np,lb2,lb,c,tt,t2,t1,b,a]) 

    # create new model and load:
    m2 = dy.Model()
    [xa,xb,xt1,xt2,xtt,xc,xlb,xlb2,xnp] = m2.load("test_new")
    #m3 = dy.Model()
    #[rnp,rlb2,rlb,rc,rtt,rt2,rt1,rb,ra] = m3.load("test_new_r")
    m3,[rnp,rlb2,rlb,rc,rtt,rt2,rt1,rb,ra] = dy.Model.from_file("test_new_r")

    # partial save and load:
    m.save("test_new_partial", [a,tt,lb2])
    m4 = dy.Model()
    [pa,ptt,plb2] = m4.load("test_new_partial")

    # types
    params = [a,xa,ra,pa,c,xc,rc]
    for p1 in params:
        assert(isinstance(p1,dy.Parameters))
    for p1 in [b,xb,rb]:
        assert(isinstance(p1,dy.LookupParameters))
    for p1 in [lb,lb2,xlb,xlb2,rlb,rlb2,plb2]:
        assert(isinstance(p1,dy.LSTMBuilder))
    for p1 in [t1,t2,xt1,xt2,rt1,rt2]:
        assert(isinstance(p1,Transfer))
    for p1 in [tt,xtt,rtt,ptt]:
        assert(isinstance(p1,MultiTransfer))
    for p1 in [np,xnp,rnp]:
        assert(isinstance(p1,NoParameters))

    # param equalities
    for p1 in [a,xa,ra,pa]:
        for p2 in [a,xa,ra,pa]:
            assert(numpy.array_equal(p1.as_array(),p2.as_array()))
    for p1 in [c,xc,rc]:
        for p2 in [c,xc,rc]:
            assert(numpy.array_equal(p1.as_array(),p2.as_array()))
    for p1 in [b,xb,rb]:
        for p2 in [b,xb,rb]:
            assert(numpy.array_equal(p1.as_array(),p2.as_array()))
    v1 = b[4]
    v2 = xb[4]
    v3 = rb[4]
    assert(numpy.array_equal(v1.value(), v2.value()))
    assert(numpy.array_equal(v1.value(), v3.value()))
    # lstm builders equalities
    s1 = lb.initial_state()
    s2 = xlb.initial_state()
    s3 = rlb.initial_state()
    y1 = s1.add_input(v1).output().value()
    y2 = s2.add_input(v1).output().value()
    y3 = s3.add_input(v1).output().value()
    for y in [y2,y3]:
        assert(numpy.array_equal(y1,y))

    # Transfer equalities
    for p1 in [t1,xt1,rt1]:
        for p2 in [t1,xt1,rt1]:
            assert(numpy.array_equal(p1.W.as_array(),p2.W.as_array()))
            assert(numpy.array_equal(p1.b.as_array(),p2.b.as_array()))
            assert(p1.nin == p2.nin)
    
    # MultiTransfer equalities
    for p1 in [tt,xtt,rtt]:
        for p2 in [tt,xtt,rtt]:
            assert(numpy.array_equal(p1.transfers[0].W.as_array(),p2.transfers[0].W.as_array()))
            assert(numpy.array_equal(p1.transfers[0].b.as_array(),p2.transfers[0].b.as_array()))
            assert(numpy.array_equal(p1.transfers[-1].W.as_array(),p2.transfers[-1].W.as_array()))
            assert(numpy.array_equal(p1.transfers[-1].b.as_array(),p2.transfers[-1].b.as_array()))
            assert(p1.transfers[0].nin == p2.transfers[0].nin)
            assert(p1.transfers[-1].nin == p2.transfers[-1].nin)

    # NoParameter equalities
    for p1 in [np,xnp,rnp]:
        assert(p1.act == dy.tanh)

    for suf in ['','.pyk','.pym']:
        os.remove("test_new"+suf)
        os.remove("test_new_r"+suf)
        os.remove("test_new_partial"+suf)

new_style_save_and_load()

print("Model saving tests passed.")




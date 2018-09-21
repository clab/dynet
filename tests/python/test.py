import dynet as dy
import numpy as np
import unittest
import gc


def npvalue_callable(x):
    return x.npvalue()


def gradient_callable(x):
    return x.gradient()


class TestInput(unittest.TestCase):

    def setUp(self):
        self.input_vals = np.arange(81)
        self.squared_norm = (self.input_vals**2).sum()
        self.shapes = [(81,), (3, 27), (3, 3, 9), (3, 3, 3, 3)]

    def test_inputTensor_not_batched(self):
        for i in range(4):
            dy.renew_cg()
            input_tensor = self.input_vals.reshape(self.shapes[i])
            x = dy.inputTensor(input_tensor)
            self.assertEqual(x.dim()[0], self.shapes[i],
                             msg="Dimension mismatch")
            self.assertEqual(x.dim()[1], 1,
                             msg="Dimension mismatch")
            self.assertTrue(
                np.allclose(x.npvalue(), input_tensor),
                msg="Expression value different from initial value"
            )
            self.assertEqual(
                dy.squared_norm(x).scalar_value(), self.squared_norm,
                msg="Value mismatch"
            )

    def test_sparse_inputTensor(self):
        dy.renew_cg()
        input_tensor = self.input_vals.reshape((3, 3, 3, 3))
        input_vals = [input_tensor[0, 0, 0, 0], input_tensor[0, 1, 2, 0]]
        input_indices = ([0, 0], [0, 1], [0, 2], [0, 0])
        x = dy.sparse_inputTensor(
            input_indices, input_vals, (3, 3, 3, 3), batched=True)
        self.assertEqual(x.dim()[0], (3, 3, 3),
                         msg="Dimension mismatch")
        self.assertEqual(x.dim()[1], 3,
                         msg="Dimension mismatch")
        self.assertTrue(np.allclose(x.npvalue()[0, 0, 0, 0], input_vals[0]),
                        msg="Expression value different from initial value")
        self.assertTrue(np.allclose(x.npvalue()[0, 1, 2, 0], input_vals[1]),
                        msg="Expression value different from initial value")
        self.assertTrue(np.allclose(x.npvalue()[1, 1, 1, 1], 0),
                        msg="Expression value different from initial value")

    def test_inputTensor_batched(self):
        for i in range(4):
            dy.renew_cg()
            input_tensor = self.input_vals.reshape(self.shapes[i])
            xb = dy.inputTensor(input_tensor, batched=True)
            self.assertEqual(
                xb.dim()[0],
                (self.shapes[i][:-1] if i > 0 else (1,)),
                msg="Dimension mismatch"
            )
            self.assertEqual(
                xb.dim()[1],
                self.shapes[i][-1],
                msg="Dimension mismatch"
            )
            self.assertTrue(
                np.allclose(xb.npvalue(), input_tensor),
                msg="Expression value different from initial value"
            )
            self.assertEqual(
                dy.sum_batches(dy.squared_norm(xb)).scalar_value(),
                self.squared_norm,
                msg="Value mismatch"
            )

    def test_inputTensor_batched_list(self):
        for i in range(4):
            dy.renew_cg()
            input_tensor = self.input_vals.reshape(self.shapes[i])
            xb = dy.inputTensor([np.asarray(x).transpose()
                                 for x in input_tensor.transpose()])
            self.assertEqual(
                xb.dim()[0],
                (self.shapes[i][:-1] if i > 0 else (1,)),
                msg="Dimension mismatch"
            )
            self.assertEqual(
                xb.dim()[1],
                self.shapes[i][-1],
                msg="Dimension mismatch"
            )
            self.assertTrue(
                np.allclose(xb.npvalue(), input_tensor),
                msg="Expression value different from initial value"
            )
            self.assertEqual(
                dy.sum_batches(dy.squared_norm(xb)).scalar_value(),
                self.squared_norm,
                msg="Value mismatch"
            )

    def test_inputTensor_except(self):
        dy.renew_cg()
        self.assertRaises(TypeError, dy.inputTensor, batched=True)


class TestParameters(unittest.TestCase):

    def setUp(self):
        # Create model
        self.m = dy.ParameterCollection()
        # Parameters
        self.p1 = self.m.add_parameters((10, 10), init=dy.ConstInitializer(1))
        self.p2 = self.m.add_parameters((10, 10), init=dy.ConstInitializer(1))
        self.lp1 = self.m.add_lookup_parameters(
            (10, 10), init=dy.ConstInitializer(1))
        self.lp2 = self.m.add_lookup_parameters(
            (10, 10), init=dy.ConstInitializer(1))
        # Trainer
        self.trainer = dy.SimpleSGDTrainer(self.m, learning_rate=0.1)
        self.trainer.set_clip_threshold(-1)

    def test_list(self):
        [p1, p2] = self.m.parameters_list()
        [lp1, lp2] = self.m.lookup_parameters_list()

    def test_shape(self):
        shape = (10, 5, 2)
        lp = self.m.add_lookup_parameters(shape)
        lp_shape = lp.shape()
        self.assertEqual(shape[0], lp_shape[0])
        self.assertEqual(shape[1], lp_shape[1])
        self.assertEqual(shape[2], lp_shape[2])

    def test_as_array(self):
        # Values
        self.p1.as_array()
        self.lp1.as_array()
        self.lp1.row_as_array(0)
        self.lp1.rows_as_array([5, 6, 9])
        # Gradients
        self.p1.grad_as_array()
        self.lp1.as_array()
        self.lp1.row_grad_as_array(0)
        self.lp1.rows_grad_as_array([5, 6, 9])

    def test_grad(self):
        # add parameter
        p = self.m.parameters_from_numpy(np.arange(5))
        # create cg
        dy.renew_cg()
        # input tensor
        x = dy.inputTensor(np.arange(5).reshape((1, 5)))
        # compute dot product
        res = x * p
        # Run forward and backward pass
        res.forward()
        res.backward()
        # Should print the value of x
        self.assertTrue(np.allclose(p.grad_as_array(),
                                    x.npvalue()), msg="Gradient is wrong")

    def test_set_value(self):
        # add parameter
        p = self.m.add_parameters((2, 3), init=dy.ConstInitializer(1))
        value_to_set = np.arange(6).reshape(2, 3)
        # set the value
        p.set_value(value_to_set)
        self.assertTrue(np.allclose(p.as_array(), value_to_set))

    def test_is_updated(self):
        self.assertTrue(self.p1.is_updated())
        self.assertTrue(self.p2.is_updated())
        self.assertTrue(self.lp1.is_updated())
        self.assertTrue(self.lp2.is_updated())

    def test_set_updated(self):
        self.p2.set_updated(False)
        self.lp1.set_updated(False)

        self.assertTrue(self.p1.is_updated())
        self.assertFalse(self.p2.is_updated())
        self.assertFalse(self.lp1.is_updated())
        self.assertTrue(self.lp2.is_updated())

        self.p1.set_updated(True)
        self.p2.set_updated(False)
        self.lp1.set_updated(False)
        self.lp2.set_updated(True)

        self.assertTrue(self.p1.is_updated())
        self.assertFalse(self.p2.is_updated())
        self.assertFalse(self.lp1.is_updated())
        self.assertTrue(self.lp2.is_updated())

        self.p1.set_updated(False)
        self.p2.set_updated(True)
        self.lp1.set_updated(True)
        self.lp2.set_updated(False)

        self.assertFalse(self.p1.is_updated())
        self.assertTrue(self.p2.is_updated())
        self.assertTrue(self.lp1.is_updated())
        self.assertFalse(self.lp2.is_updated())

        dy.renew_cg()

        a = self.p1 * self.lp1[1]
        b = self.p2 * self.lp2[1]
        loss = dy.dot_product(a, b) / 100
        loss.backward()

        self.trainer.update()

        ones = np.ones((10, 10))
        self.assertTrue(np.allclose(self.p1.as_array(), ones),
                        msg=np.array_str(self.p1.as_array()))
        self.assertTrue(np.allclose(self.lp2.as_array()[1], ones[
                        0]), msg=np.array_str(self.lp2.as_array()))

    def test_update(self):
        ones = np.ones((10, 10))

        dy.renew_cg()

        a = self.p1 * self.lp1[1]
        b = self.p2 * self.lp2[1]
        loss = dy.dot_product(a, b) / 100

        self.assertEqual(loss.scalar_value(), 10, msg=str(loss.scalar_value()))

        loss.backward()

        # Check the gradients
        self.assertTrue(np.allclose(self.p1.grad_as_array(), 0.1 * ones),
                        msg=np.array_str(self.p1.grad_as_array()))
        self.assertTrue(np.allclose(self.p2.grad_as_array(), 0.1 * ones),
                        msg=np.array_str(self.p2.grad_as_array()))
        self.assertTrue(np.allclose(self.lp1.grad_as_array()[1], ones[
                        0]), msg=np.array_str(self.lp1.grad_as_array()))
        self.assertTrue(np.allclose(self.lp2.grad_as_array()[1], ones[
                        0]), msg=np.array_str(self.lp2.grad_as_array()))

        self.trainer.update()

        # Check the updated parameters
        self.assertTrue(np.allclose(self.p1.as_array(), ones * 0.99),
                        msg=np.array_str(self.p1.as_array()))
        self.assertTrue(np.allclose(self.p2.as_array(), ones * 0.99),
                        msg=np.array_str(self.p2.as_array()))
        self.assertTrue(np.allclose(self.lp1.as_array()[1], ones[
                        0] * 0.9), msg=np.array_str(self.lp1.as_array()[1]))
        self.assertTrue(np.allclose(self.lp2.as_array()[1], ones[
                        0] * 0.9), msg=np.array_str(self.lp2.as_array()))

    def test_param_change_after_update(self):
        for trainer_type in dy.SimpleSGDTrainer, dy.AdamTrainer:
            trainer = trainer_type(self.m)
            for _ in range(100):
                p = self.m.add_parameters((1,))
                dy.renew_cg()
                p.forward()
                p.backward()
                trainer.update()

    def test_delete_model(self):
        p = dy.ParameterCollection().add_parameters(
            (1,), init=dy.ConstInitializer(1)
        )
        p.value()
        gc.collect()
        p.value()

    def test_delete_parent_model(self):
        model = dy.ParameterCollection().add_subcollection()
        p = model.add_parameters(
            (1,), init=dy.ConstInitializer(1)
        )
        p.value()
        gc.collect()
        p.value()

    def test_parameters_initializers(self):

        self.m.add_parameters((3, 5), init=0)
        self.m.add_parameters((3, 5), init='uniform', scale=2.0)
        self.m.add_parameters((3, 5), init='normal', mean=-1.0, std=2.5)
        self.m.add_parameters((5, 5), init='identity')
        # self.m.add_parameters((5,5), init='saxe')
        self.m.add_parameters((3, 5), init='glorot')
        self.m.add_parameters((3, 5), init='he')
        arr = np.zeros((3, 5))
        self.m.add_parameters(arr.shape, init=arr)
        self.m.add_parameters((3, 5), init=dy.ConstInitializer(2.0))

    def test_lookup_parameters_initializers(self):

        p = self.m.add_lookup_parameters((3, 5), init=0)
        p = self.m.add_lookup_parameters((3, 5), init='uniform', scale=2.0)
        p = self.m.add_lookup_parameters(
            (3, 5), init='normal', mean=-1.0, std=2.5)
        p = self.m.add_lookup_parameters((3, 5), init='glorot')
        p = self.m.add_lookup_parameters((3, 5), init='he')
        arr = np.zeros((3, 5))
        p = self.m.add_lookup_parameters(arr.shape, init=arr)
        p = self.m.add_lookup_parameters((3, 5), init=dy.ConstInitializer(2.0))

        array = np.arange(50).reshape(10, 5)
        p = self.m.add_lookup_parameters(array.shape, init=array)

        slice_array = array[8]
        slice_param = p.batch([8]).npvalue()

        for i in range(5):
            self.assertEqual(slice_array[i], slice_param[i])


class TestBatchManipulation(unittest.TestCase):

    def setUp(self):
        # create model
        self.m = dy.ParameterCollection()
        # Parameter
        self.p = self.m.add_lookup_parameters((2, 3))
        # Values
        self.pval = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        self.p.init_from_array(self.pval)

    def test_lookup_batch(self):
        dy.renew_cg()
        x = dy.lookup_batch(self.p, [0, 1])
        self.assertTrue(np.allclose(x.npvalue(), self.pval.T))

    def test_pick_batch_elem(self):
        dy.renew_cg()
        x = dy.lookup_batch(self.p, [0, 1])
        y = dy.pick_batch_elem(x, 1)
        self.assertTrue(np.allclose(y.npvalue(), self.pval[1]))

    def test_pick_batch_elems(self):
        dy.renew_cg()
        x = dy.lookup_batch(self.p, [0, 1])
        y = dy.pick_batch_elems(x, [0])
        self.assertTrue(np.allclose(y.npvalue(), self.pval[0]))
        z = dy.pick_batch_elems(x, [0, 1])
        self.assertTrue(np.allclose(z.npvalue(), self.pval.T))

    def test_concatenate_to_batch(self):
        dy.renew_cg()
        x = dy.lookup_batch(self.p, [0, 1])
        y = dy.pick_batch_elem(x, 0)
        z = dy.pick_batch_elem(x, 1)
        w = dy.concatenate_to_batch([y, z])
        self.assertTrue(np.allclose(w.npvalue(), self.pval.T))


class TestIOPartialWeightDecay(unittest.TestCase):
    def setUp(self):
        self.file = "tmp.model"
        self.m = dy.ParameterCollection()
        self.m2 = dy.ParameterCollection()
        self.p = self.m.add_parameters(1)
        self.t = dy.SimpleSGDTrainer(self.m)

    def test_save_load(self):
        self.p.forward()
        self.p.backward()
        self.t.update()
        dy.renew_cg()
        v1 = self.p.value()
        dy.save(self.file, [self.p])
        [p2] = dy.load(self.file, self.m2)
        v2 = p2.value()
        self.assertTrue(np.allclose(v1, v2))


class TestIOEntireModel(unittest.TestCase):
    def setUp(self):
        self.file = "bilstm.model"
        self.m = dy.ParameterCollection()
        self.m2 = dy.ParameterCollection()
        self.b = dy.BiRNNBuilder(2, 10, 10, self.m, dy.LSTMBuilder)
        # Custom parameters
        self.W1 = self.m.add_parameters(10)
        self.W2 = self.m.add_parameters(12)

    def test_save_load(self):
        self.m.save(self.file)
        dy.BiRNNBuilder(2, 10, 10, self.m2, dy.LSTMBuilder)
        self.m2.add_parameters(10)
        self.m2.add_parameters(12)
        self.m2.populate(self.file)

    def test_save_load_with_gradient(self):
        # Make it so W1 has a gradient
        dy.renew_cg()
        dy.sum_elems(self.W1).backward()
        # Record gradients
        W1_grad = self.W1.grad_as_array()
        W2_grad = self.W2.grad_as_array()
        # Save the ParameterCollection
        self.m.save(self.file)
        # Populate
        self.m.populate(self.file)
        # Check that the gradients were saved
        self.assertTrue(np.allclose(self.W1.grad_as_array(), W1_grad))
        self.assertTrue(np.allclose(self.W2.grad_as_array(), W2_grad))



class TestIOPartial(unittest.TestCase):
    def setUp(self):
        self.file = "tmp.model"
        self.m = dy.ParameterCollection()
        self.m2 = dy.ParameterCollection()
        self.L = self.m.add_lookup_parameters((10, 2), name="la")
        self.a = self.m.add_parameters(10, name="a")

    def test_save_load(self):
        self.L.save(self.file, "/X")
        self.a.save(self.file, append=True)
        a = self.m2.add_parameters(10)
        L = self.m2.add_lookup_parameters((10, 2))
        L.populate(self.file, "/X")
        a.populate(self.file, "/a")


class TestIOHighLevelAPI(unittest.TestCase):

    def setUp(self):
        self.file = "bilstm.model"
        # create models
        self.m = dy.ParameterCollection()
        self.m2 = dy.ParameterCollection()
        # Create birnn
        self.b = dy.BiRNNBuilder(2, 10, 10, self.m, dy.LSTMBuilder)

    def test_save_load(self):
        dy.save(self.file, [self.b])
        [b] = dy.load(self.file, self.m2)

    def test_save_load_generator(self):
        dy.save(self.file, (x for x in [self.b]))
        [b] = list(dy.load_generator(self.file, self.m2))


class TestExpression(unittest.TestCase):

    def setUp(self):

        self.v1 = np.arange(10)
        self.v2 = np.arange(10)[::-1]

    def test_value(self):
        dy.renew_cg()
        x = dy.inputTensor(self.v1)
        self.assertTrue(np.allclose(x.npvalue(), self.v1))

    def test_value_sanity(self):
        dy.renew_cg()
        x = dy.inputTensor(self.v1)
        dy.renew_cg()
        self.assertRaises(RuntimeError, npvalue_callable, x)

    def test_gradient(self):
        dy.renew_cg()
        x = dy.inputTensor(self.v1)
        y = dy.inputTensor(self.v2)
        loss = dy.dot_product(x, y)
        loss.forward()
        loss.backward(full=True)
        self.assertTrue(np.allclose(x.gradient(), self.v2),
                        msg="{}\n{}\n{}\n{}\n{}".format(
                        loss.value(),
                        x.gradient(),
                        self.v2,
                        y.gradient(),
                        self.v2
                        ))

    def test_gradient_sanity(self):
        dy.renew_cg()
        x = dy.inputTensor(self.v1)
        y = dy.inputTensor(self.v2)
        loss = dy.dot_product(x, y)
        loss.forward()
        self.assertRaises(RuntimeError, gradient_callable, x)


class TestOperations(unittest.TestCase):

    def setUp(self):
        # create model
        self.m = dy.ParameterCollection()
        self.v1 = np.arange(10)
        self.v2 = np.arange(10)
        self.v3 = np.arange(10)

    def test_layer_norm(self):
        dy.renew_cg()
        x = dy.inputTensor(self.v1)
        g = dy.inputTensor(self.v2)
        b = dy.inputTensor(self.v3)
        y = dy.layer_norm(x, g, b)
        loss = dy.sum_elems(y)

        loss.backward()

        centered_v1 = self.v1 - self.v1.mean()
        y_np_value = self.v2 / self.v1.std() * centered_v1 + self.v3

        self.assertTrue(np.allclose(y.npvalue(), y_np_value))


class TestSlicing(unittest.TestCase):

    def test_slicing(self):
        dy.renew_cg()
        data = np.random.random((10, 10, 10))
        self.assertTrue(np.allclose(dy.inputTensor(
            data)[:1, :2, :3].npvalue(), data[:1, :2, :3]))
        self.assertTrue(np.allclose(dy.inputTensor(data, batched=True)[
                        :1, :2, :3].npvalue(), data[:1, :2, :3]))
        self.assertTrue(np.allclose(dy.inputTensor(
            data)[:, :, :3].npvalue(), data[:, :, :3]))
        self.assertTrue(np.allclose(dy.inputTensor(
            data)[3:, :, :].npvalue(), data[3:, :, :]))
        self.assertTrue(np.allclose(dy.inputTensor(
            data)[:, :, ::1].npvalue(), data[:, :, ::1]))
        self.assertTrue(np.allclose(dy.inputTensor(
            data)[:, :, ::3].npvalue(), data[:, :, ::3]))
        self.assertTrue(np.allclose(dy.inputTensor(
            data)[3:5, 1:3, 1:].npvalue(), data[3:5, 1:3, 1:]))


class TestSimpleRNN(unittest.TestCase):

    def setUp(self):
        # create model
        self.m = dy.ParameterCollection()
        self.rnn = dy.SimpleRNNBuilder(2, 10, 10, self.m)

    def test_get_parameters(self):
        dy.renew_cg()
        self.rnn.initial_state()
        P_p = self.rnn.get_parameters()
        P_e = self.rnn.get_parameter_expressions()
        for l_p, l_e in zip(P_p, P_e):
            for w_p, w_e in zip(l_p, l_e):
                self.assertTrue(np.allclose(w_e.npvalue(), w_p.as_array()))

    def test_get_parameters_sanity(self):
        self.assertRaises(
            ValueError, lambda x: x.get_parameter_expressions(), self.rnn)


class TestGRU(unittest.TestCase):

    def setUp(self):
        # create model
        self.m = dy.ParameterCollection()
        self.rnn = dy.GRUBuilder(2, 10, 10, self.m)

    def test_get_parameters(self):
        dy.renew_cg()
        self.rnn.initial_state()
        P_p = self.rnn.get_parameters()
        P_e = self.rnn.get_parameter_expressions()
        for l_p, l_e in zip(P_p, P_e):
            for w_p, w_e in zip(l_p, l_e):
                self.assertTrue(np.allclose(w_e.npvalue(), w_p.as_array()))

    def test_get_parameters_sanity(self):
        self.assertRaises(
            ValueError, lambda x: x.get_parameter_expressions(), self.rnn)


class TestVanillaLSTM(unittest.TestCase):

    def setUp(self):
        # create model
        self.m = dy.ParameterCollection()
        self.rnn = dy.VanillaLSTMBuilder(2, 10, 10, self.m)

    def test_get_parameters(self):
        dy.renew_cg()
        self.rnn.initial_state()
        P_p = self.rnn.get_parameters()
        P_e = self.rnn.get_parameter_expressions()
        for l_p, l_e in zip(P_p, P_e):
            for w_p, w_e in zip(l_p, l_e):
                self.assertTrue(np.allclose(w_e.npvalue(), w_p.as_array()))

    def test_get_parameters_sanity(self):
        self.assertRaises(
            ValueError, lambda x: x.get_parameter_expressions(), self.rnn)

    def test_initial_state_vec(self):
        dy.renew_cg()
        init_s = [dy.ones(10), dy.ones(10), dy.ones(10), dy.ones(10)]
        self.rnn.initial_state(init_s)

    def test_set_h(self):
        dy.renew_cg()
        init_h = [dy.ones(10), dy.ones(10)]
        state = self.rnn.initial_state()
        state.set_h(init_h)

    def test_set_c(self):
        dy.renew_cg()
        init_c = [dy.ones(10), dy.ones(10)]
        state = self.rnn.initial_state()
        state.set_s(init_c)

    def test_set_s(self):
        dy.renew_cg()
        init_s = [dy.ones(10), dy.ones(10), dy.ones(10), dy.ones(10)]
        state = self.rnn.initial_state()
        state.set_s(init_s)


class TestCoupledLSTM(unittest.TestCase):

    def setUp(self):
        # create model
        self.m = dy.ParameterCollection()
        self.rnn = dy.CoupledLSTMBuilder(2, 10, 10, self.m)

    def test_get_parameters(self):
        dy.renew_cg()
        self.rnn.initial_state()
        P_p = self.rnn.get_parameters()
        P_e = self.rnn.get_parameter_expressions()
        for l_p, l_e in zip(P_p, P_e):
            for w_p, w_e in zip(l_p, l_e):
                self.assertTrue(np.allclose(w_e.npvalue(), w_p.as_array()))

    def test_get_parameters_sanity(self):
        self.assertRaises(
            ValueError, lambda x: x.get_parameter_expressions(), self.rnn)

    def test_initial_state_vec(self):
        dy.renew_cg()
        init_s = [dy.ones(10), dy.ones(10), dy.ones(10), dy.ones(10)]
        self.rnn.initial_state(init_s)

    def test_set_h(self):
        dy.renew_cg()
        init_h = [dy.ones(10), dy.ones(10)]
        state = self.rnn.initial_state()
        state.set_h(init_h)

    def test_set_c(self):
        dy.renew_cg()
        init_c = [dy.ones(10), dy.ones(10)]
        state = self.rnn.initial_state()
        state.set_s(init_c)

    def test_set_s(self):
        dy.renew_cg()
        init_s = [dy.ones(10), dy.ones(10), dy.ones(10), dy.ones(10)]
        state = self.rnn.initial_state()
        state.set_s(init_s)

class TestSparseLSTM(unittest.TestCase):

    def setUp(self):
        # create model
        self.m = dy.ParameterCollection()
        self.rnn = dy.SparseLSTMBuilder(2, 10, 10, self.m)

    def test_get_parameters(self):
        dy.renew_cg()
        self.rnn.initial_state()
        P_p = self.rnn.get_parameters()
        P_e = self.rnn.get_parameter_expressions()
        for l_p, l_e in zip(P_p, P_e):
            for w_p, w_e in zip(l_p, l_e):
                self.assertTrue(np.allclose(w_e.npvalue(), w_p.as_array()))

    def test_get_parameters_sanity(self):
        self.assertRaises(ValueError, lambda x: x.get_parameter_expressions(), self.rnn)

class TestFastLSTM(unittest.TestCase):

    def setUp(self):
        # create model
        self.m = dy.ParameterCollection()
        self.rnn = dy.FastLSTMBuilder(2, 10, 10, self.m)

    def test_get_parameters(self):
        dy.renew_cg()
        self.rnn.initial_state()
        P_p = self.rnn.get_parameters()
        P_e = self.rnn.get_parameter_expressions()
        for l_p, l_e in zip(P_p, P_e):
            for w_p, w_e in zip(l_p, l_e):
                self.assertTrue(np.allclose(w_e.npvalue(), w_p.as_array()))

    def test_get_parameters_sanity(self):
        self.assertRaises(
            ValueError, lambda x: x.get_parameter_expressions(), self.rnn)

    def test_initial_state_vec(self):
        dy.renew_cg()
        init_s = [dy.ones(10), dy.ones(10), dy.ones(10), dy.ones(10)]
        self.rnn.initial_state(init_s)

    def test_set_h(self):
        dy.renew_cg()
        init_h = [dy.ones(10), dy.ones(10)]
        state = self.rnn.initial_state()
        state.set_h(init_h)

    def test_set_c(self):
        dy.renew_cg()
        init_c = [dy.ones(10), dy.ones(10)]
        state = self.rnn.initial_state()
        state.set_s(init_c)

    def test_set_s(self):
        dy.renew_cg()
        init_s = [dy.ones(10), dy.ones(10), dy.ones(10), dy.ones(10)]
        state = self.rnn.initial_state()
        state.set_s(init_s)


class TestStandardSoftmax(unittest.TestCase):

    def setUp(self):
        # create model
        self.pc = dy.ParameterCollection()
        self.sm = dy.StandardSoftmaxBuilder(3, 10, self.pc, True)

    def test_sanity(self):
        for i in range(3):
            dy.renew_cg()
            nll = self.sm.neg_log_softmax(
                dy.inputTensor(np.arange(3)), 4, update=True)
            nll_const = self.sm.neg_log_softmax(
                dy.inputTensor(np.arange(3)), 5, update=False)
            nll = self.sm.neg_log_softmax(
                dy.inputTensor(np.arange(3)), 6, update=True)
            nll_const = self.sm.neg_log_softmax(
                dy.inputTensor(np.arange(3)), 7, update=False)
            nll.value()
            nll_const.value()


class TestClassFactoredSoftmax(unittest.TestCase):

    def setUp(self):
        # create model
        self.pc = dy.ParameterCollection()
        dic = dict()
        with open('cluster_file.txt', 'w+') as f:
            for i in range(5):
                f.write(str(i) + " " + str(2 * i) + "\n")
                f.write(str(i) + " " + str(2 * i + 1) + "\n")
                dic[str(2 * i)] = len(dic)
                dic[str(2 * i + 1)] = len(dic)
        self.sm = dy.ClassFactoredSoftmaxBuilder(
            3, 'cluster_file.txt', dic, self.pc, True)

    def test_sanity(self):
        for i in range(3):
            dy.renew_cg()
            nll = self.sm.neg_log_softmax(
                dy.inputTensor(np.arange(3)), 4, update=True)
            nll_const = self.sm.neg_log_softmax(
                dy.inputTensor(np.arange(3)), 5, update=False)
            nll = self.sm.neg_log_softmax(
                dy.inputTensor(np.arange(3)), 6, update=True)
            nll_const = self.sm.neg_log_softmax(
                dy.inputTensor(np.arange(3)), 7, update=False)
            nll.value()
            nll_const.value()


if __name__ == '__main__':
    unittest.main()

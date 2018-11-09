import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers.core import Layer
import numpy as np



learning_rate = 0.0005
global_seed = 809
np.random.seed(global_seed)



class DenseLayerwithBN(tl.layers.Layer):
    def __init__(
            self,
            prev_layer,
            n_units=100,
            is_train=False,
            bn=False,
            W_init=tf.truncated_normal_initializer(stddev=0.1, seed=global_seed),
            b_init=tf.constant_initializer(value=0.0),
            name='Dense_with_bn'
    ):
        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs
        self.is_train = is_train

        n_in = int(self.inputs.get_shape()[-1])  # obtain pre_layer's input number
        with tf.variable_scope(name):
            W = tf.get_variable(name='W', shape=(n_in, n_units), initializer=W_init, dtype=tf.float32)
            b = tf.get_variable(name='b', shape=n_units, initializer=b_init, dtype=tf.float32)
            w_x_b = tf.matmul(self.inputs, W) + b
            if bn:
                print("DenseLayer(bn)  %s: %d %s" % (self.name, n_units, "bn"))
                w_x_b = tf.layers.batch_normalization(w_x_b, training=self.is_train, name='norm')
            else:
                print("DenseLayer  %s: %d" % (self.name, n_units))
            self.outputs = tf.nn.relu(w_x_b)

        # update layer paras
        self.all_layers.append(self.outputs)
        self.all_params.extend([W, b])


class DNNRegressor:
    """The :class:`DNNRegressor` Regressor based on full_connected layers' DNN.
    It's flexible and Highly customizable. You can customize the network input size,
    the number of hidden layer neurons and the number of hidden layers,
    you can also choose whether to add batch norm or dropout layer.

    Parameters
    ----------
    sess : A tensorflow's `Session` object
        A tenorflow session to run the network.
    feature_size : int
        The number of units of Input.
    hidden : list
         A list to define hidden layers, each integer of the list indicates the number of neurons of each layer.
         for example:
            hidden = [128,128,64], indicates that there are 3 hidden layers in the network, and the number of hidden
            layer neurons is 128, 64, 32 respectively.
    bn : boolean
        Using batch norm or not.
        If True, batch norm will be added before activation.
    drop_out : boolean
        Adding Drop_out layer or not.
        If True, drop_out layer will be added between hidden layers.
    """
    def __init__(self, sess, feature_size, hidden, bn=False, drop_out=False):
        self.sess = sess
        self.feature_size = feature_size
        self.hidden = hidden
        self.bn = bn
        self.drop_out = drop_out
        self.is_train = False
        self.X_input = tf.placeholder(tf.float32, shape=[None, feature_size], name="X_input")
        self.y_input = tf.placeholder(tf.float32, shape=[None, 1], name="y_input")

        self.y_pred, self.paras, self.network = self.create_network()

        self.loss = tl.cost.mean_squared_error(self.y_pred, self.y_input)

        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=self.paras)

        tl.layers.initialize_global_variables(self.sess)


    def create_network(self):
        input_x = self.X_input
        network = tl.layers.InputLayer(inputs=input_x, name='input_layer')
        ly = 0
        for n_unit in self.hidden:
            ly += 1
            network = DenseLayerwithBN(network, n_units=n_unit,
                                       is_train=self.is_train, bn=self.bn, name="Dense_bn"+str(ly))
            if self.drop_out:
                network = tl.layers.DropoutLayer(network, keep=0.8, seed=global_seed, name='drop'+str(ly))
        network = tl.layers.DenseLayer(network, n_units=1, act=tf.identity)

        return network.outputs, network.all_params, network

    def predict(self, X):
        """
        :param X: the input of dnn
        :return: regressor's prediction
        """
        self.is_train = False
        # For testing, disable dropout as follow.
        feed_dict = {self.X_input: X}
        dp_dict = tl.utils.dict_to_one(self.network.all_drop)
        feed_dict.update(dp_dict)
        y_pred = self.sess.run(self.y_pred, feed_dict=feed_dict)
        return y_pred

    def train(self, X, y, batch_size=32, n_epoch=2):
        """
        :param X: train samples' X
        :param y: train samples' label y
        :param batch_size: mini batch size
        :param n_epoch: training epoches
        """
        self.is_train = True
        step = 0

        for epoch in range(n_epoch):
            # print("****** train_epoch: %d " % epoch)
            batch_num = len(y)//batch_size+1
            for i in range(batch_num):
                batch_indices = np.random.randint(0, len(y), batch_size)
                X_train_a = X[batch_indices]
                y_train_a = y[batch_indices]
                feed_dict = {self.X_input: X_train_a, self.y_input: y_train_a}
                feed_dict.update(self.network.all_drop)
                self.sess.run(self.optimizer, feed_dict=feed_dict)
                step += 1



if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    from sklearn.datasets import load_boston

    boston = load_boston()
    X = boston.data
    y = boston.target
    y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=6)

    # Standard Scaler
    ss_X = StandardScaler()
    ss_y = StandardScaler()

    X_train_ss = ss_X.fit_transform(X_train)
    X_test_ss = ss_X.transform(X_test)

    y_train_ss = ss_y.fit_transform(y_train)
    y_test_ss = ss_y.transform(y_test)

    # get a dnn regressor
    sess = tf.InteractiveSession()
    dnn = DNNRegressor(sess=sess, feature_size=X.shape[-1], hidden=[128, 128, 128, 64, 64], bn=False, drop_out=True)

    # train the regressor
    dnn.train(X_train_ss, y_train_ss, batch_size=32, n_epoch=1000)

    # use the regressor to predict test sets
    y_pred_ss = dnn.predict(X_test_ss)
    y_pred = ss_y.inverse_transform(y_pred_ss)


    MSE_test = mean_squared_error(y_test, y_pred)
    print("Mean squared error(test): %.4f" % MSE_test)

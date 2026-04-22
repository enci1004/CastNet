class CoupledBiLSTM(layers.Layer):
    def __init__(self, units, return_sequences=False, **kwargs):
        super(CoupledBiLSTM, self).__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences

        self.backward_cell = layers.LSTMCell(units)
        self.forward_cell = layers.LSTMCell(units)

    def build(self, input_shape):
        super().build(input_shape)
        input_dim = input_shape[-1]

        self.W_mod = self.add_weight(
            name='modulation_weights',
            shape=(self.units, 4 * self.units),
            initializer='glorot_uniform'
        )

        self.forward_cell.build((None, input_dim))

        self.forward_kernel = self.forward_cell.kernel
        self.forward_recurrent_kernel = self.forward_cell.recurrent_kernel
        self.forward_bias = self.forward_cell.bias

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        timesteps = tf.shape(inputs)[1]
        T = timesteps

        h_backward = tf.zeros((batch_size, self.units))
        c_backward = tf.zeros((batch_size, self.units))
        backward_states = tf.TensorArray(tf.float32, size=T)

        for t in tf.range(T - 1, -1, -1):
            x_backward = inputs[:, t, :]
            h_backward, (_, c_backward) = self.backward_cell(
                x_backward, [h_backward, c_backward]
            )
            backward_states = backward_states.write(t, h_backward)

        backward_sequence = tf.transpose(backward_states.stack(), [1, 0, 2])

        h_forward = tf.zeros((batch_size, self.units))
        c_forward = tf.zeros((batch_size, self.units))
        forward_outputs = tf.TensorArray(tf.float32, size=T)

        for t in range(T):
            x_forward = inputs[:, t, :]
            b_t = backward_sequence[:, t, :]
            h_prev = h_forward

            inputs_part = tf.matmul(x_forward, self.forward_kernel)
            hidden_part = tf.matmul(h_prev, self.forward_recurrent_kernel)
            z = inputs_part + hidden_part + self.forward_bias

            modulation = tf.matmul(b_t, self.W_mod)
            z += modulation

            z0, z1, z2, z3 = tf.split(z, 4, axis=-1)

            i = tf.sigmoid(z0)
            f = tf.sigmoid(z1)
            o = tf.sigmoid(z2)
            c_candidate = tf.tanh(z3)

            c_forward = f * c_forward + i * c_candidate
            h_forward = o * tf.tanh(c_forward)

            forward_outputs = forward_outputs.write(t, h_forward)

        forward_sequence = tf.transpose(forward_outputs.stack(), [1, 0, 2])

        combined_sequence = tf.concat(
            [forward_sequence, backward_sequence],
            axis=-1
        )

        if self.return_sequences:
            return combined_sequence
        else:
            return combined_sequence[:, -1, :]

    def get_config(self):
        config = super(CoupledBiLSTM, self).get_config()
        config.update({
            'units': self.units,
            'return_sequences': self.return_sequences
        })
        return config



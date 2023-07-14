import tensorflow as tf
import rsqtoa.fapprox.rsqtoa_fapprox_config as config

class CustomCallback(tf.keras.callbacks.Callback):
    """ Callback Object, which implements the callback functions tf calls during training """
    def __init__(self, monitored_key, epsilon):
        super(CustomCallback, self).__init__()
        self.epsilon = epsilon
        self.monitored_key = monitored_key

    def on_epoch_end(self, epoch, logs=None):
        """ tell the model to stop training if self.monitored_key reached self.epsilon """
        #IMPROVE: stop if we reached it consistently
        if logs[self.monitored_key] < self.epsilon:
            self.model.stop_training = True
            print("Finished training early: threshhold epsilon < {}".format(self.epsilon))

class FunctionApproximator(tf.keras.models.Sequential):
    def __init__(self):
        super(FunctionApproximator, self).__init__()
        """ training loss and accuracy aggregators. All these are aggregators which give the mean on .results()"""

        # for the trained model
        self.train_loss_points = tf.keras.metrics.Mean(name='train_loss_points')
        self.train_loss_gradients = tf.keras.metrics.Mean(name='train_loss_gradients')
        self.train_loss_all = tf.keras.metrics.Mean(name='train_loss_all')

        self.test_loss_points = tf.keras.metrics.Mean(name='test_loss_points')
        self.test_loss_gradients = tf.keras.metrics.Mean(name='test_loss_gradients')
        self.test_loss_all = tf.keras.metrics.Mean(name='test_loss_all')
        # deprecated. but shape important!
        # TODO: Look at convert_dict to plottable, to see how these have to look in the end!
        """
        self.lossDict = {"train_loss_points": [], "train_loss_gradients": [], "train_loss_all": [],
                         "test_loss_points": [], "test_loss_gradients": [], "test_loss_all": [] }
        """

    def history_to_loss_dict(self, history_dict):
        lossDict = {}
        for key in history_dict:
            lossDict[key] = [list(range(len(history_dict[key]))), history_dict[key]]
        return lossDict

    def compile_and_fit(self, train_dataset, test_dataset):
        # Compile model
        self.optimizer = config.optimizer
        self.compile(optimizer=config.optimizer, loss=config.loss)


        # to test_loss_points, a 'val' is appended to the original name, because it's not taken by validation!
        self.earlyStoppingCallback = CustomCallback("val_test_loss_points", 1e-5)

        # Train model
        history = self.fit(
          train_dataset, validation_data=test_dataset,
          epochs=config.epochs, callbacks=[self.earlyStoppingCallback])

        return self.history_to_loss_dict(history.history)

    @tf.function
    def rms_gradient_loss(self, ground_truth_gradients, predicted_gradients):
        """ Root mean square on gradients:
        * gradients are shape (?, n_grad_entries). Take diff_norm = euclidean_norm(gt-pred)
        * Then take rmse of all diffs_norms!
        """

        diff = ground_truth_gradients-predicted_gradients   # shape (?, n_grad_entries)
        diff_norm = tf.norm(diff, axis = 1)                 # shape (?, )

        # aggregate along batch dimension (?, ) => ()
        return tf.sqrt(tf.reduce_mean(tf.pow(diff_norm,2)))

    @tf.function
    def rms_point_loss(self, ground_truth_ys, predicted_ys):
        """ Root mean square on ys. Note ys are shape (?, 1)"""
        diff = tf.cast(ground_truth_ys[:,0], tf.float32) - predicted_ys[:,0]
        return tf.sqrt(tf.reduce_mean(tf.pow(diff,2)))

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        return [self.train_loss_points, self.train_loss_gradients, self.train_loss_all,
                self.test_loss_points, self.test_loss_gradients, self.test_loss_all]

    def train_step(self, inputs):
      if config.learn_gradients:
        return self.train_step_gradients(inputs)
      else:
        return self.train_step_no_gradients(inputs)

    def test_step(self, inputs):
      if config.learn_gradients:
        return self.test_step_gradients(inputs)
      else:
        return self.test_step_no_gradients(inputs)

    @tf.function
    def train_step_gradients(self, inputs):
        xs, ys, ys_dxs = inputs
        gradient_factor = self.beta
        #model = self
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xs)
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self(xs, training=True)

            point_loss = self.rms_point_loss(ys, predictions)

            predicted_gradients = tape.gradient(predictions, xs)

            gradients_loss = tf.cast(self.rms_gradient_loss(ys_dxs, predicted_gradients),tf.float32)

            lossTerm = (point_loss + gradient_factor*gradients_loss)/(1+gradient_factor)

        optimizer_gradients = tape.gradient(lossTerm, self.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars=zip(optimizer_gradients, self.trainable_variables))

        self.train_loss_points(point_loss)
        self.train_loss_gradients(gradients_loss)
        # if interesting, log this too
        self.train_loss_all(lossTerm)
        return {"train_loss_points" : self.train_loss_points.result(),
                "train_loss_gradients" : self.train_loss_gradients.result(),
                "train_loss_all" : self.train_loss_all.result()}

    @tf.function
    def train_step_no_gradients(self, inputs):
        xs, ys = inputs # TODO: this _ has to be removed
        #gradient_factor = self.beta

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xs)
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self(xs, training=True)

            point_loss = self.rms_point_loss(ys, predictions)

        optimizer_gradients = tape.gradient(point_loss, self.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars=zip(optimizer_gradients, self.trainable_variables))

        self.train_loss_points(point_loss)
        return {"train_loss_points": self.train_loss_points.result()}

    @tf.function
    def test_step_gradients(self, inputs):
        """ test_step is called in evaluation (on test/val-set) """ # NOTE: self is the model!
        xs, ys, ys_dxs = inputs
        gradient_factor = self.beta
        with tf.GradientTape(persistent=True) as tape:

            tape.watch(xs)
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            #predictions = model(xs, training=False)
            predictions = self(xs)

            point_loss = self.rms_point_loss(ys, predictions)

            predicted_gradients = tape.gradient(predictions, xs)

            gradients_loss = tf.cast(self.rms_gradient_loss(ys_dxs, predicted_gradients), tf.float32)
            lossTerm = (point_loss + gradient_factor * gradients_loss) / (1 + gradient_factor)


        self.test_loss_points(point_loss)
        self.test_loss_gradients(gradients_loss)
        self.test_loss_all(lossTerm)

        # note that these are called as validation data. Thus, it's adding a 'val_' to the keys!
        # TODO: I think we can keep the keys from before, and tf will automatically rename!
        return {"test_loss_points": self.test_loss_points.result(),
                "test_loss_gradients": self.test_loss_gradients.result(),
                "test_loss_all": self.test_loss_all.result()}

    @tf.function
    def test_step_no_gradients(self, inputs):
        """ test_step is called in evaluation (on test/val-set) """ # NOTE: self is the model!
        xs, ys = inputs
        with tf.GradientTape(persistent=True) as tape:

            tape.watch(xs)
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            #predictions = model(xs, training=False)
            predictions = self(xs)
            point_loss = self.rms_point_loss(ys, predictions)

        self.test_loss_points(point_loss)

        # note that these are called as validation data. Thus, it's adding a 'val_' to the keys!
        # TODO: I think we can keep the keys from before, and tf will automatically rename!
        return {"test_loss_points": self.test_loss_points.result()}

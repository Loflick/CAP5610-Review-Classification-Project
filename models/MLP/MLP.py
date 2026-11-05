"""MLP Model for Review Classification"""

import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

EMBED_DIM = 300
NUM_CLASSES = 5

# Default experiment vars
DEFAULT_CONFIG = {
    "hidden_sizes": (256, 128),
    "activation": "relu",
    "learning_rate": 1e-3,
    "batch_size": 256,
    "epochs": 10,
    "weight_power": 0.5,
}


class MLP:
    """MLP: input, hidden layers, output."""

    def __init__(
        self,
        hidden_sizes=(256, 128),
        activation="relu",
        learning_rate=1e-3,
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
    ):
        # Init model hyperparameters
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.model = self.build()

    def build(self):
        """Builds the MLP model."""
        model = Sequential(name="mlp")

        # Input layer
        model.add(Input(shape=(self.embed_dim,), name="input"))

        # Hidden layers
        # z = sum(w_i * x_i) + b
        # ReLU: f(z) = max(0, z)
        for n, size in enumerate(self.hidden_sizes, start=1):
            model.add(Dense(size, activation=self.activation, name=f"hidden_{n}"))

        # Output layer
        # Softmax: y_hat_i = e^(z_i) / sum_j(e^(z_j))
        model.add(Dense(self.num_classes, activation="softmax", name="output"))

        # Optimizer and loss
        # L = -(1 / N) * sum(log(y_hat_i))
        # Gradient step: w = w - eta * dL/dw
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def fit(self, x_train, y_train, x_val, y_val, batch_size=256, epochs=10, class_weight=None):
        """Train, validating after each epoch."""
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            class_weight=class_weight,
        )

        return history

    def predict(self, x):
        """Return the predicted class for each item."""
        probs = self.model.predict(x, batch_size=1024, verbose=0)
        return np.argmax(probs, axis=1)

"""LSTM-Autoencoder als sequenzbasierter Detektor (TICKET-02, optionale ``dl``-Dependency).

Operiert auf **Fenstern** (3D: ``(n_windows, window, n_features)``, vgl.
``automl_ad.data.make_windows`` / ``load_windowed``) und erfüllt das PyOD-artige Interface
(``fit`` / ``decision_function`` / ``decision_scores_`` / ``threshold_``). Score = mittlerer
Rekonstruktionsfehler eines Fensters (höher = anomaler). Erfasst die zeitliche Dynamik des
Prozesses (z. B. driftende Fehler wie IDV 13).

Wird via Auto-Discovery (``detectors/base.py``) als ``"lstm_ae"`` registriert. **Hinweis:**
Anders als die tabellarischen Detektoren erwartet dieser Detektor **3D**-Sequenzeingaben.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .. import config


class _LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, window: int, hidden: int = 64, latent: int = 16):
        super().__init__()
        self.window = window
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.enc2lat = nn.Linear(hidden, latent)
        self.lat2hid = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, F)
        _, (h, _) = self.encoder(x)
        z = self.enc2lat(h[-1])                                  # (B, latent)
        dec_in = self.lat2hid(z).unsqueeze(1).repeat(1, self.window, 1)
        dec_out, _ = self.decoder(dec_in)                        # (B, T, hidden)
        return self.out(dec_out)                                 # (B, T, F)


class LSTMAEDetector:
    """LSTM-Autoencoder mit PyOD-artigem Interface (auf Sequenzfenstern)."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden: int = 64,
        epoch_num: int = 30,
        lr: float = 1e-3,
        batch_size: int = 256,
        contamination: float = config.DEFAULT_CONTAMINATION,
        random_state: int = config.RANDOM_SEED,
        device: str | None = None,
    ):
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.epoch_num = epoch_num
        self.lr = lr
        self.batch_size = batch_size
        self.contamination = contamination
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _check_3d(X: np.ndarray) -> None:
        if np.ndim(X) != 3:
            raise ValueError(
                "LSTM-AE erwartet 3D-Sequenzen (n_windows, window, n_features). "
                "Nutze automl_ad.data.make_windows / load_windowed."
            )

    def _recon_error(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        errors = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = torch.tensor(X[i : i + self.batch_size], dtype=torch.float32, device=self.device)
                rec = self.model_(xb)
                err = ((rec - xb) ** 2).mean(dim=(1, 2))      # je Fenster
                errors.append(err.cpu().numpy())
        return np.concatenate(errors) if errors else np.empty(0)

    def fit(self, X: np.ndarray, y=None) -> "LSTMAEDetector":
        self._check_3d(X)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        n_features, window = X.shape[2], X.shape[1]
        self.model_ = _LSTMAutoencoder(n_features, window, self.hidden, self.latent_dim).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        n = len(X_t)
        self.model_.train()
        for _ in range(self.epoch_num):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                xb = X_t[perm[i : i + self.batch_size]].to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model_(xb), xb)
                loss.backward()
                opt.step()

        self.decision_scores_ = self._recon_error(X)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1 - self.contamination))
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_3d(X)
        return self._recon_error(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) > self.threshold_).astype(int)


def make_lstm_ae(**hp) -> LSTMAEDetector:
    return LSTMAEDetector(**hp)


FACTORIES = {"lstm_ae": make_lstm_ae}

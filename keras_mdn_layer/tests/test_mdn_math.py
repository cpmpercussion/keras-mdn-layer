"""Mathematical correctness tests for the MDN loss function.

These tests verify the implementation against analytically computable values
from well-known probability results, ensuring the loss function correctly
implements the negative log-likelihood of a Gaussian mixture model.

References:
    - Bishop, C. M. (1994). "Mixture Density Networks." NCRG/94/004.
    - Bishop, C. M. (2006). "Pattern Recognition and Machine Learning", Ch. 5.6.
"""

import keras
import keras_mdn_layer as mdn
import numpy as np
import math
import pytest


# ---------------------------------------------------------------------------
# 1. Single-component analytical NLL verification
# ---------------------------------------------------------------------------

class TestSingleGaussianNLL:
    """Verify loss against the closed-form NLL of a single Gaussian.

    For K=1 mixture with D-dimensional diagonal-covariance normal:
        NLL = D/2 log(2 pi) + sum_d log(sigma_d) + 1/2 sum_d ((y_d - mu_d)/sigma_d)^2

    The mixing coefficient contribution vanishes since log softmax([z]) = 0
    for any scalar z.
    """

    def _analytical_nll_1d(self, y, mu, sigma):
        """Analytical NLL for a 1D Gaussian: 0.5*log(2*pi) + log(sigma) + 0.5*((y-mu)/sigma)^2"""
        return 0.5 * math.log(2.0 * math.pi) + math.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2

    def _analytical_nll_nd(self, y, mu, sigma):
        """Analytical NLL for a D-dimensional diagonal Gaussian."""
        y, mu, sigma = np.asarray(y), np.asarray(mu), np.asarray(sigma)
        D = len(y)
        return (0.5 * D * math.log(2.0 * math.pi)
                + np.sum(np.log(sigma))
                + 0.5 * np.sum(((y - mu) / sigma) ** 2))

    def _make_pred(self, mus, sigmas, pi_logits):
        """Pack mu, sigma, pi_logits into y_pred tensor (batch_size=1)."""
        return np.concatenate([mus, sigmas, pi_logits], axis=-1).astype(np.float32).reshape(1, -1)

    def test_standard_normal_at_zero(self):
        """NLL of y=0 under N(0,1): should be 0.5*log(2*pi) ≈ 0.9189."""
        loss_func = mdn.get_mixture_loss_func(1, 1)
        y_true = np.array([[0.0]], dtype=np.float32)
        y_pred = self._make_pred([0.0], [1.0], [0.0])

        expected = self._analytical_nll_1d(0.0, 0.0, 1.0)
        actual = float(loss_func(y_true, y_pred))
        np.testing.assert_allclose(actual, expected, rtol=1e-5,
                                   err_msg="NLL of y=0 under N(0,1) should be 0.5*log(2*pi)")

    def test_standard_normal_at_one(self):
        """NLL of y=1 under N(0,1): should be 0.5*log(2*pi) + 0.5 ≈ 1.4189."""
        loss_func = mdn.get_mixture_loss_func(1, 1)
        y_true = np.array([[1.0]], dtype=np.float32)
        y_pred = self._make_pred([0.0], [1.0], [0.0])

        expected = self._analytical_nll_1d(1.0, 0.0, 1.0)
        actual = float(loss_func(y_true, y_pred))
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_scaled_sigma(self):
        """NLL with sigma=2: wider distribution, different normalisation."""
        loss_func = mdn.get_mixture_loss_func(1, 1)
        y_true = np.array([[0.0]], dtype=np.float32)
        y_pred = self._make_pred([0.0], [2.0], [0.0])

        expected = self._analytical_nll_1d(0.0, 0.0, 2.0)
        actual = float(loss_func(y_true, y_pred))
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_shifted_mean(self):
        """NLL with mu=3, y=5, sigma=1."""
        loss_func = mdn.get_mixture_loss_func(1, 1)
        y_true = np.array([[5.0]], dtype=np.float32)
        y_pred = self._make_pred([3.0], [1.0], [0.0])

        expected = self._analytical_nll_1d(5.0, 3.0, 1.0)
        actual = float(loss_func(y_true, y_pred))
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_2d_output(self):
        """NLL for a 2D diagonal Gaussian."""
        loss_func = mdn.get_mixture_loss_func(2, 1)
        y = [1.0, -1.0]
        mu = [0.5, 0.5]
        sigma = [1.0, 2.0]
        y_true = np.array([y], dtype=np.float32)
        y_pred = self._make_pred(mu, sigma, [0.0])

        expected = self._analytical_nll_nd(y, mu, sigma)
        actual = float(loss_func(y_true, y_pred))
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_5d_output(self):
        """NLL for a 5D diagonal Gaussian with varied parameters."""
        D = 5
        loss_func = mdn.get_mixture_loss_func(D, 1)
        np.random.seed(123)
        y = np.random.randn(D)
        mu = np.random.randn(D)
        sigma = np.abs(np.random.randn(D)) + 0.1
        y_true = np.array([y], dtype=np.float32)
        y_pred = self._make_pred(mu.tolist(), sigma.tolist(), [0.0])

        expected = self._analytical_nll_nd(y, mu, sigma)
        actual = float(loss_func(y_true, y_pred))
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_pi_logit_value_irrelevant_for_single_mix(self):
        """For K=1, any pi_logit value should give the same loss (softmax([z])=1 for all z)."""
        loss_func = mdn.get_mixture_loss_func(1, 1)
        y_true = np.array([[2.0]], dtype=np.float32)

        losses = []
        for logit in [-100.0, -1.0, 0.0, 1.0, 100.0]:
            y_pred = self._make_pred([1.0], [1.0], [logit])
            losses.append(float(loss_func(y_true, y_pred)))

        np.testing.assert_allclose(losses, losses[0], rtol=1e-5,
                                   err_msg="Single-mix loss should not depend on pi_logit value")


# ---------------------------------------------------------------------------
# 2. Multi-component mixture verification
# ---------------------------------------------------------------------------

class TestMixtureNLL:
    """Verify mixture loss against manually computed mixture log-probabilities.

    For K identical components with equal mixing weights:
        p(y) = sum_k (1/K) * N(y|mu,sigma^2) = N(y|mu,sigma^2)
    so the NLL should equal the single-component NLL.

    For K different components, we compute analytically:
        p(y) = sum_k pi_k * N(y|mu_k, sigma_k^2)
        NLL = -log(p(y))
    """

    def _gaussian_pdf_1d(self, y, mu, sigma):
        """1D Gaussian PDF."""
        return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((y - mu) / sigma) ** 2)

    def test_identical_components_equal_single(self):
        """K identical components with equal weights should equal single-component NLL."""
        K = 3
        loss_func_mix = mdn.get_mixture_loss_func(1, K)
        loss_func_single = mdn.get_mixture_loss_func(1, 1)

        mu, sigma = 2.0, 1.5
        y = 3.0

        y_true = np.array([[y]], dtype=np.float32)
        # All K components have same mu, sigma; equal logits -> equal weights
        y_pred_mix = np.concatenate([[mu] * K, [sigma] * K, [0.0] * K]).astype(np.float32).reshape(1, -1)
        y_pred_single = np.array([[mu, sigma, 0.0]], dtype=np.float32)

        loss_mix = float(loss_func_mix(y_true, y_pred_mix))
        loss_single = float(loss_func_single(y_true, y_pred_single))
        np.testing.assert_allclose(loss_mix, loss_single, rtol=1e-5,
                                   err_msg="Identical components should give same NLL as single component")

    def test_two_component_mixture_analytical(self):
        """Verify 2-component mixture NLL against hand computation.

        pi = [0.5, 0.5] (equal logits), mu = [0, 4], sigma = [1, 1], y = 2
        p(y) = 0.5 * N(2|0,1) + 0.5 * N(2|4,1)
        """
        loss_func = mdn.get_mixture_loss_func(1, 2)
        y = 2.0
        y_true = np.array([[y]], dtype=np.float32)
        # equal logits [0,0] -> pi = [0.5, 0.5]
        y_pred = np.array([[0.0, 4.0, 1.0, 1.0, 0.0, 0.0]], dtype=np.float32)

        p = 0.5 * self._gaussian_pdf_1d(y, 0.0, 1.0) + 0.5 * self._gaussian_pdf_1d(y, 4.0, 1.0)
        expected_nll = -math.log(p)

        actual = float(loss_func(y_true, y_pred))
        np.testing.assert_allclose(actual, expected_nll, rtol=1e-5)

    def test_unequal_weights_analytical(self):
        """Verify mixture with unequal weights.

        logits = [1, 0] -> pi = [e/(e+1), 1/(e+1)] ≈ [0.731, 0.269]
        mu = [0, 10], sigma = [1, 1], y = 0
        """
        loss_func = mdn.get_mixture_loss_func(1, 2)
        y = 0.0
        y_true = np.array([[y]], dtype=np.float32)
        y_pred = np.array([[0.0, 10.0, 1.0, 1.0, 1.0, 0.0]], dtype=np.float32)

        # compute mixture weights from softmax
        pi_0 = math.exp(1.0) / (math.exp(1.0) + math.exp(0.0))
        pi_1 = math.exp(0.0) / (math.exp(1.0) + math.exp(0.0))
        p = pi_0 * self._gaussian_pdf_1d(y, 0.0, 1.0) + pi_1 * self._gaussian_pdf_1d(y, 10.0, 1.0)
        expected_nll = -math.log(p)

        actual = float(loss_func(y_true, y_pred))
        np.testing.assert_allclose(actual, expected_nll, rtol=1e-4)

    def test_batch_loss_is_mean_of_per_sample(self):
        """Loss over a batch should be the mean of per-sample losses."""
        loss_func = mdn.get_mixture_loss_func(1, 2)

        y_true_1 = np.array([[1.0]], dtype=np.float32)
        y_true_2 = np.array([[3.0]], dtype=np.float32)
        y_true_batch = np.array([[1.0], [3.0]], dtype=np.float32)

        y_pred_row = np.array([[0.0, 2.0, 1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        y_pred_batch = np.tile(y_pred_row, (2, 1))

        loss_1 = float(loss_func(y_true_1, y_pred_row))
        loss_2 = float(loss_func(y_true_2, y_pred_row))
        loss_batch = float(loss_func(y_true_batch, y_pred_batch))

        expected = (loss_1 + loss_2) / 2.0
        np.testing.assert_allclose(loss_batch, expected, rtol=1e-5,
                                   err_msg="Batch loss should be mean of individual losses")


# ---------------------------------------------------------------------------
# 3. Softmax correctness
# ---------------------------------------------------------------------------

class TestSoftmaxCorrectness:
    """Verify the numpy softmax against known identities."""

    def test_matches_scipy_style(self):
        """Compare with direct numpy computation of softmax."""
        logits = [2.0, 1.0, 0.1, -1.0]
        e = np.exp(np.array(logits) - np.max(logits))
        expected = e / e.sum()
        actual = mdn.softmax(logits)
        np.testing.assert_allclose(actual, expected, rtol=1e-7)

    def test_temperature_limit_cold(self):
        """As temp -> 0, softmax should approach one-hot on the max."""
        logits = [1.0, 3.0, 2.0]
        result = mdn.softmax(logits, t=1e-6)
        np.testing.assert_allclose(result[1], 1.0, atol=1e-6)

    def test_temperature_limit_hot(self):
        """As temp -> inf, softmax should approach uniform."""
        logits = [1.0, 3.0, 2.0]
        result = mdn.softmax(logits, t=1e6)
        np.testing.assert_allclose(result, 1.0 / 3.0, atol=1e-4)

    def test_invariant_to_constant_shift(self):
        """softmax(x) == softmax(x + c) for any constant c."""
        logits = [1.0, 2.0, 3.0]
        r1 = mdn.softmax(logits)
        r2 = mdn.softmax([x + 1000.0 for x in logits])
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


# ---------------------------------------------------------------------------
# 4. Sampling statistical tests
# ---------------------------------------------------------------------------

class TestSamplingCorrectness:
    """Verify that sampling produces outputs with correct statistics.

    For a single Gaussian component, samples should have the specified
    mean and standard deviation (within statistical tolerance).
    """

    def test_numpy_sampling_mean_and_std(self):
        """Samples from a single 1D Gaussian should have correct mean and std."""
        np.random.seed(42)
        output_dim = 1
        num_mixes = 1
        mu_val = 5.0
        sigma_val = 2.0
        params = np.array([mu_val, sigma_val, 0.0])

        samples = np.array([mdn.sample_from_output(params, output_dim, num_mixes)[0]
                            for _ in range(5000)])

        # sigma_temp=1.0, so cov = sigma^2 * I, std = sigma
        np.testing.assert_allclose(np.mean(samples), mu_val, atol=0.15,
                                   err_msg="Sample mean should approximate mu")
        np.testing.assert_allclose(np.std(samples), sigma_val, atol=0.15,
                                   err_msg="Sample std should approximate sigma")

    def test_numpy_sampling_2d_mean(self):
        """Samples from a single 2D Gaussian should have correct mean vector."""
        np.random.seed(42)
        output_dim = 2
        num_mixes = 1
        mu = [3.0, -2.0]
        sigma = [1.0, 1.0]
        params = np.array(mu + sigma + [0.0])

        samples = np.array([mdn.sample_from_output(params, output_dim, num_mixes)
                            for _ in range(5000)])

        np.testing.assert_allclose(np.mean(samples, axis=0), mu, atol=0.15)

    def test_keras_sampling_shape_and_finite(self):
        """Keras-graph sampling function should produce finite samples of correct shape."""
        output_dim = 3
        num_mixes = 5
        sampling_func = mdn.get_mixture_sampling_fun(output_dim, num_mixes)

        batch_size = 16
        mus = np.zeros((batch_size, num_mixes * output_dim))
        sigmas = np.ones((batch_size, num_mixes * output_dim))
        pi_logits = np.zeros((batch_size, num_mixes))
        y_pred = np.concatenate([mus, sigmas, pi_logits], axis=-1).astype(np.float32)

        samples = np.array(sampling_func(y_pred))
        assert samples.shape == (batch_size, output_dim)
        assert np.all(np.isfinite(samples))

    def test_mixture_sampling_bimodal(self):
        """Samples from a two-component mixture should cluster near both means."""
        np.random.seed(42)
        output_dim = 1
        num_mixes = 2
        # Two well-separated components with equal weight, small sigma
        mu1, mu2 = -5.0, 5.0
        sigma = 0.5
        params = np.array([mu1, mu2, sigma, sigma, 0.0, 0.0])

        samples = np.array([mdn.sample_from_output(params, output_dim, num_mixes)[0]
                            for _ in range(2000)])

        # Check samples near each mode
        near_mu1 = np.sum(np.abs(samples - mu1) < 2.0)
        near_mu2 = np.sum(np.abs(samples - mu2) < 2.0)
        assert near_mu1 > 400, f"Expected ~1000 samples near mu1={mu1}, got {near_mu1}"
        assert near_mu2 > 400, f"Expected ~1000 samples near mu2={mu2}, got {near_mu2}"
        # Very few samples should be in the middle
        near_zero = np.sum(np.abs(samples) < 1.0)
        assert near_zero < 50, f"Expected very few samples near 0, got {near_zero}"


# ---------------------------------------------------------------------------
# 5. Training convergence tests
# ---------------------------------------------------------------------------

class TestTrainingConvergence:
    """Test that training converges and learns correct parameters for known distributions."""

    def test_learns_single_gaussian(self):
        """Train on data from N(0,1) and verify the model learns approximately correct parameters."""
        np.random.seed(42)
        keras.utils.set_random_seed(42)

        N_MIXES = 1
        OUTPUT_DIM = 1
        N_SAMPLES = 500

        model = keras.Sequential([
            keras.layers.Dense(15, input_shape=(1,), activation='relu'),
            keras.layers.Dense(15, activation='relu'),
            mdn.MDN(OUTPUT_DIM, N_MIXES)
        ])
        model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIM, N_MIXES),
                      optimizer=keras.optimizers.Adam(learning_rate=0.005))

        # Training data: constant input -> N(0,1) output
        x_train = np.ones((N_SAMPLES, 1), dtype=np.float32)
        y_train = np.random.randn(N_SAMPLES, 1).astype(np.float32)

        history = model.fit(x_train, y_train, epochs=60, batch_size=64, verbose=0)

        # Loss should converge near the theoretical minimum: 0.5*log(2*pi*e) ≈ 1.4189
        # (the differential entropy of N(0,1))
        final_loss = history.history['loss'][-1]
        theoretical_min = 0.5 * math.log(2 * math.pi * math.e)
        assert final_loss < theoretical_min + 0.5, \
            f"Final loss {final_loss:.3f} should be near theoretical min {theoretical_min:.3f}"

        # Check that model predictions give reasonable mu and sigma
        pred = model.predict(np.array([[1.0]], dtype=np.float32), verbose=0)[0]
        mu, sigma, pi = mdn.split_mixture_params(pred, OUTPUT_DIM, N_MIXES)
        np.testing.assert_allclose(mu[0], 0.0, atol=0.5,
                                   err_msg="Learned mu should be near 0")
        np.testing.assert_allclose(sigma[0], 1.0, atol=0.5,
                                   err_msg="Learned sigma should be near 1")

    def test_learns_shifted_gaussian(self):
        """Train on data from N(5, 0.5^2) and verify learned parameters."""
        np.random.seed(123)
        keras.utils.set_random_seed(123)

        N_MIXES = 1
        OUTPUT_DIM = 1
        TRUE_MU = 5.0
        TRUE_SIGMA = 0.5

        model = keras.Sequential([
            keras.layers.Dense(15, input_shape=(1,), activation='relu'),
            keras.layers.Dense(15, activation='relu'),
            mdn.MDN(OUTPUT_DIM, N_MIXES)
        ])
        model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIM, N_MIXES),
                      optimizer=keras.optimizers.Adam(learning_rate=0.005))

        x_train = np.ones((500, 1), dtype=np.float32)
        y_train = (np.random.randn(500, 1) * TRUE_SIGMA + TRUE_MU).astype(np.float32)

        model.fit(x_train, y_train, epochs=80, batch_size=64, verbose=0)

        pred = model.predict(np.array([[1.0]], dtype=np.float32), verbose=0)[0]
        mu, sigma, pi = mdn.split_mixture_params(pred, OUTPUT_DIM, N_MIXES)
        np.testing.assert_allclose(mu[0], TRUE_MU, atol=0.5,
                                   err_msg=f"Learned mu should be near {TRUE_MU}")
        np.testing.assert_allclose(sigma[0], TRUE_SIGMA, atol=0.5,
                                   err_msg=f"Learned sigma should be near {TRUE_SIGMA}")

    def test_loss_monotonically_trends_down(self):
        """Over enough epochs, the smoothed loss should decrease."""
        np.random.seed(42)
        keras.utils.set_random_seed(42)

        model = keras.Sequential([
            keras.layers.Dense(15, input_shape=(1,), activation='relu'),
            mdn.MDN(1, 3)
        ])
        model.compile(loss=mdn.get_mixture_loss_func(1, 3),
                      optimizer=keras.optimizers.Adam(learning_rate=0.01))

        x = np.random.uniform(-1, 1, (200, 1)).astype(np.float32)
        y = (x ** 2).astype(np.float32)

        history = model.fit(x, y, epochs=20, batch_size=32, verbose=0)
        losses = history.history['loss']

        # Compare mean of first quarter to mean of last quarter
        q = len(losses) // 4
        early_mean = np.mean(losses[:q])
        late_mean = np.mean(losses[-q:])
        assert late_mean < early_mean, "Smoothed loss should decrease over training"


# ---------------------------------------------------------------------------
# 6. Numerical stability edge cases
# ---------------------------------------------------------------------------

class TestNumericalStability:
    """Test that the loss function handles edge cases without NaN/Inf."""

    def test_very_small_sigma(self):
        """Loss should remain finite even with very small sigma."""
        loss_func = mdn.get_mixture_loss_func(1, 1)
        y_true = np.array([[0.0]], dtype=np.float32)
        # sigma very small but positive (ELU+1+eps guarantees this in practice)
        y_pred = np.array([[0.0, 1e-6, 0.0]], dtype=np.float32)
        loss = float(loss_func(y_true, y_pred))
        assert np.isfinite(loss)

    def test_very_large_sigma(self):
        """Loss should remain finite with large sigma."""
        loss_func = mdn.get_mixture_loss_func(1, 1)
        y_true = np.array([[0.0]], dtype=np.float32)
        y_pred = np.array([[0.0, 1e4, 0.0]], dtype=np.float32)
        loss = float(loss_func(y_true, y_pred))
        assert np.isfinite(loss)

    def test_extreme_logits(self):
        """Loss should remain finite with extreme pi logits (log_softmax handles this)."""
        loss_func = mdn.get_mixture_loss_func(1, 3)
        y_true = np.array([[0.0]], dtype=np.float32)
        # One logit dominates
        y_pred = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 100.0, -100.0, -100.0]], dtype=np.float32)
        loss = float(loss_func(y_true, y_pred))
        assert np.isfinite(loss)

    def test_large_distance_from_mean(self):
        """Loss should remain finite when y is far from all means."""
        loss_func = mdn.get_mixture_loss_func(1, 2)
        y_true = np.array([[1000.0]], dtype=np.float32)
        y_pred = np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        loss = float(loss_func(y_true, y_pred))
        assert np.isfinite(loss)


# ---------------------------------------------------------------------------
# 7. End-to-end inference pipeline
# ---------------------------------------------------------------------------

class TestInferencePipeline:
    """Test the full predict -> split -> sample pipeline."""

    def test_predict_split_sample_roundtrip(self):
        """Build a model, predict, split params, and sample — all should work."""
        OUTPUT_DIM = 2
        N_MIXES = 3
        np.random.seed(42)

        model = keras.Sequential([
            keras.layers.Dense(10, input_shape=(1,), activation='relu'),
            mdn.MDN(OUTPUT_DIM, N_MIXES)
        ])
        model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIM, N_MIXES), optimizer='adam')

        x = np.array([[1.0]], dtype=np.float32)
        y_pred = model.predict(x, verbose=0)
        assert y_pred.shape == (1, 2 * OUTPUT_DIM * N_MIXES + N_MIXES)

        params = y_pred[0]
        mus, sigmas, pi_logits = mdn.split_mixture_params(params, OUTPUT_DIM, N_MIXES)
        assert len(mus) == N_MIXES * OUTPUT_DIM
        assert len(sigmas) == N_MIXES * OUTPUT_DIM
        assert len(pi_logits) == N_MIXES

        # sigmas should be positive (ELU+1+eps activation)
        assert np.all(sigmas > 0), "Sigmas should always be positive"

        # pi_logits -> valid probability distribution
        pis = mdn.softmax(pi_logits)
        np.testing.assert_allclose(pis.sum(), 1.0, rtol=1e-5)

        sample = mdn.sample_from_output(params, OUTPUT_DIM, N_MIXES)
        assert sample.shape == (OUTPUT_DIM,)
        assert np.all(np.isfinite(sample))

    def test_trained_model_sampling_matches_data(self):
        """After training, samples from the model should match the training distribution."""
        np.random.seed(42)
        keras.utils.set_random_seed(42)

        OUTPUT_DIM = 1
        N_MIXES = 1
        TRUE_MU = 3.0

        model = keras.Sequential([
            keras.layers.Dense(15, input_shape=(1,), activation='relu'),
            keras.layers.Dense(15, activation='relu'),
            mdn.MDN(OUTPUT_DIM, N_MIXES)
        ])
        model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIM, N_MIXES),
                      optimizer=keras.optimizers.Adam(learning_rate=0.005))

        x_train = np.ones((300, 1), dtype=np.float32)
        y_train = (np.random.randn(300, 1) * 0.5 + TRUE_MU).astype(np.float32)
        model.fit(x_train, y_train, epochs=60, batch_size=64, verbose=0)

        # Generate samples from the trained model
        pred = model.predict(np.array([[1.0]], dtype=np.float32), verbose=0)[0]
        samples = np.array([mdn.sample_from_output(pred, OUTPUT_DIM, N_MIXES)[0]
                            for _ in range(1000)])

        np.testing.assert_allclose(np.mean(samples), TRUE_MU, atol=0.5,
                                   err_msg="Samples from trained model should be near training mean")

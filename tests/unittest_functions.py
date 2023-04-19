from absl.testing import absltest
from functools import partial

import numpy as np
import numpy.typing
import jax
import jax.numpy as jnp

import rowan
from rowan.functions import _promote_vec as rowan_promote_vec

import jax_quaternion as jqt


class TestQuaternionMath(absltest.TestCase):
    def test_exp(self):
        # Random Matrix of Quaternions:
        v = np.random.rand(10, 4)

        # Exponential Quaternion Function:
        result_1 = jqt.exp(v)
        result_2 = jqt.exp(v[0, :])

        # Rowan Result:
        true_value = rowan.exp(v)

        # Assertion Test:
        np.testing.assert_allclose(
            true_value, result_1,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            true_value[0, :], result_2,
            atol=1e-6,
        )

    def test_multiply(self):
        qi = np.random.rand(10, 4)
        qj = np.random.rand(10, 4)
        result = jqt.multiply(qi, qj)

        # Rowan Result:
        true_value = rowan.multiply(qi, qj)

        # Assertion Test:
        np.testing.assert_allclose(
            true_value, result,
            atol=1e-6,
        )

    def test_promote_vector(self):
        v = np.random.rand(10, 3)
        result = jqt._jax_promote_vector(v)

        # Rowan Result:
        true_value = rowan_promote_vec(v)

        # Assertion Test:
        np.testing.assert_allclose(
            true_value, result,
            atol=1e-6,
        )

    def test_integrate(self):
        q = np.random.rand(10, 4)
        q = q / np.linalg.norm(q, axis=-1)[..., np.newaxis]
        v = np.random.rand(10, 3)
        dt = 0.1
        result = jqt.integrate(q, v, dt)

        # Rowan Result:
        true_value = rowan.calculus.integrate(q, v, dt)

        # Assertion Test:
        np.testing.assert_allclose(
            true_value, result,
            atol=1e-6,
        )

    def test_normalize(self):
        q = np.random.rand(10, 4)
        result_1 = jqt.normalize(q)
        result_2 = jqt._numpy_normalize(q)

        # Rowan Result:
        true_value = rowan.normalize(q)

        # Assertion Test:
        np.testing.assert_allclose(
            true_value, result_1,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            true_value, result_2,
            atol=1e-6,
        )

    def test_conjugate(self):
        q = np.random.rand(10, 4)
        result = jqt.conjugate(q)

        # Rowan Result:
        true_value = rowan.conjugate(q)

        # Assertion Test:
        np.testing.assert_allclose(
            true_value, result,
            atol=1e-6,
        )

    def test_rotation(self):
        q = np.random.rand(10, 4)
        v = np.random.rand(10, 3)
        result = jqt.rotate(q, v)

        # Rowan Result:
        true_value = rowan. rotate(q, v)

        # Assertion Test:
        np.testing.assert_allclose(
            true_value, result,
            atol=1e-6,
        )


if __name__ == "__main__":
    absltest.main()

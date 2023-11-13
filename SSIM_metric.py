import tensorflow as tf
import numpy as np

def ciede2000(x, y, k_L=1, k_C=1, k_H=1):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    # Conversion to Lab color space
    L = x[:, 0]
    a = x[:, 1]
    b = x[:, 2]
    C = tf.sqrt(a * a + b * b)

    Lp = y[:, 0]
    ap = y[:, 1]
    bp = y[:, 2]
    Cp = tf.sqrt(ap * ap + bp * bp)

    L_mean = 0.5 * (L + Lp)
    C_mean = 0.5 * (C + Cp)

    G = 0.5 * (1 - tf.sqrt(C_mean ** 7 / (C_mean ** 7 + 25 ** 7)))
    a_prime = (1 + G) * a
    a_prime_p = (1 + G) * ap
    C_prime = tf.sqrt(a_prime ** 2 + b ** 2)
    C_prime_p = tf.sqrt(a_prime_p ** 2 + bp ** 2)

    h_prime = tf.atan2(b, a_prime)
    h_prime += (h_prime < 0) * (2 * np.pi)
    h_prime_p = tf.atan2(bp, a_prime_p)
    h_prime_p += (h_prime_p < 0) * (2 * np.pi)

    delta_L = Lp - L
    delta_C = C_prime_p - C_prime

    h_diff = h_prime_p - h_prime
    h_diff -= (h_diff > np.pi) * (2 * np.pi)
    h_diff += (h_diff < -np.pi) * (2 * np.pi)

    delta_H = 2 * tf.sqrt(C_prime_p * C_prime) * tf.sin(h_diff / 2.0)

    L_mean_p = 0.5 * (Lp + L)
    C_mean_p = 0.5 * (C_prime + C_prime_p)

    T = 1 - 0.17 * tf.cos(h_prime - np.pi / 6) + 0.24 * tf.cos(2 * h_prime) + 0.32 * tf.cos(3 * h_prime + np.pi / 30) - 0.20 * tf.cos(4 * h_prime - 63 * np.pi / 180)

    delta_theta = np.pi / 6 * tf.exp(-((180 / np.pi * h_prime - 275) / 25) ** 2)

    R_C = 2 * tf.sqrt(C_mean_p ** 7 / (C_mean_p ** 7 + 25 ** 7))
    S_L = 1 + (0.015 * (L_mean_p - 50) ** 2) / tf.sqrt(20 + (L_mean_p - 50) ** 2)
    S_C = 1 + 0.045 * C_mean_p
    S_H = 1 + 0.015 * C_mean_p * T
    R_T = -tf.sin(2 * delta_theta) * R_C

    delta_E = tf.sqrt(
        (delta_L / (k_L * S_L)) ** 2
        + (delta_C / (k_C * S_C)) ** 2
        + (delta_H / (k_H * S_H)) ** 2
        + R_T * (delta_C / (k_C * S_C)) * (delta_H / (k_H * S_H))
    )

    return delta_E

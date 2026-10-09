# scripts/test_bindings.py
import numpy as np
import minivec_cpp


def test_l2_binding():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 6.0], dtype=np.float32)
    d = float(minivec_cpp.l2(a, b))
    assert abs(d - 3.0) < 1e-6
    print("PASS: l2 =", d)

if __name__ == "__main__":
    test_l2_binding()

# scripts/test_bindings.py
import numpy as np
import minivec_cpp
def test_l2():
    a = np.array([1.0,2.0,3.0], dtype=np.float32)
    b = np.array([1.0,2.0,6.0], dtype=np.float32)
    d = float(minivec_cpp.l2_distance(a,b))
    assert abs(d - 3.0) < 1e-6
    print("PASS: l2_distance =", d)

if __name__ == "__main__":
    test_l2()

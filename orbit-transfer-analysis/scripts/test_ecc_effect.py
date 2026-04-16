import sys
sys.path.append("/Users/heewon/Desktop/무제 폴더")

from orbit_transfer.pipeline.evaluate import create_database, evaluate_transfer

db = create_database()

cases = [
    {"name": "circular", "e0": 0.0, "ef": 0.0},
    {"name": "eccentric", "e0": 0.0503, "ef": 0.078784},
]

for case in cases:
    label, result = evaluate_transfer(
        h0=400.0,
        delta_a=-2.491105,
        delta_i=13.847986,
        T_normed=0.279623,
        e0=case["e0"],
        ef=case["ef"],
        db=db,
    )

    print(f"\n[{case['name']}]")
    print("e0 =", case["e0"])
    print("ef =", case["ef"])
    print("label =", label)
    print("converged =", result.converged)
    print("cost =", result.cost)
    print("n_peaks =", result.n_peaks)
    print("profile_class =", result.profile_class)
    print("nu0 =", result.nu0)
    print("nuf =", result.nuf)
    print("T_f =", result.t[-1])
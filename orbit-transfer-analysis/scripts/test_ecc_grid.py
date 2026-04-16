import sys
sys.path.append("/Users/heewon/Desktop/무제 폴더")

from orbit_transfer.pipeline.evaluate import create_database, evaluate_transfer

db = create_database()

cases = [
    # baseline circular
    {
        "name": "case_1_circular_small_di",
        "h0": 400.0,
        "delta_a": -2.491105,
        "delta_i": 3.0,
        "T_normed": 0.279623,
        "e0": 0.0,
        "ef": 0.0,
    },
    {
        "name": "case_2_ecc_small_di",
        "h0": 400.0,
        "delta_a": -2.491105,
        "delta_i": 3.0,
        "T_normed": 0.279623,
        "e0": 0.0503,
        "ef": 0.078784,
    },

    # larger inclination change
    {
        "name": "case_3_circular_large_di",
        "h0": 400.0,
        "delta_a": -2.491105,
        "delta_i": 13.847986,
        "T_normed": 0.279623,
        "e0": 0.0,
        "ef": 0.0,
    },
    {
        "name": "case_4_ecc_large_di",
        "h0": 400.0,
        "delta_a": -2.491105,
        "delta_i": 13.847986,
        "T_normed": 0.279623,
        "e0": 0.0503,
        "ef": 0.078784,
    },

    # longer transfer time
    {
        "name": "case_5_circular_longT",
        "h0": 400.0,
        "delta_a": -2.491105,
        "delta_i": 13.847986,
        "T_normed": 1.0,
        "e0": 0.0,
        "ef": 0.0,
    },
    {
        "name": "case_6_ecc_longT",
        "h0": 400.0,
        "delta_a": -2.491105,
        "delta_i": 13.847986,
        "T_normed": 1.0,
        "e0": 0.0503,
        "ef": 0.078784,
    },
]

for case in cases:
    print("\n" + "=" * 70)
    print(case["name"])
    print("=" * 70)

    label, result = evaluate_transfer(
        h0=case["h0"],
        delta_a=case["delta_a"],
        delta_i=case["delta_i"],
        T_normed=case["T_normed"],
        e0=case["e0"],
        ef=case["ef"],
        db=db,
    )

    print("inputs:")
    print("  h0        =", case["h0"])
    print("  delta_a   =", case["delta_a"])
    print("  delta_i   =", case["delta_i"])
    print("  T_normed  =", case["T_normed"])
    print("  e0        =", case["e0"])
    print("  ef        =", case["ef"])

    print("outputs:")
    print("  label         =", label)
    print("  converged     =", result.converged)
    print("  cost          =", result.cost)
    print("  n_peaks       =", result.n_peaks)
    print("  profile_class =", result.profile_class)
    print("  nu0           =", result.nu0)
    print("  nuf           =", result.nuf)
    print("  T_f           =", result.t[-1])

db.close()
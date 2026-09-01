"""Format constants taken from ``docs/00-formal-model.md``. Do not invent extras."""

# NVFP4 / TE (formal model §1.1)
E2M1_MAX = 6.0
E4M3_MAX = 448.0
# 448 * 6 = 2688: PTS peak-to-2688 (formal model §1.1; eval plan §1)
PTS_PEAK = E4M3_MAX * E2M1_MAX
NVFP4_GROUP = 16

# E2M1 positive grid (formal model §1.1)
E2M1_POS_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# E4M3 (TE / formal model §1.1): bias 7, max finite 448, NaN at exp=all-ones & mant=all-ones
E4M3_BIAS = 7
E4M3_NAN_CODE = 0x7F  # 0b0_1111_111; sign may also be set

# HiF4 (formal model §2)
HIF4_GROUP = 64
HIF4_INTRA_MAX = 7.0  # 2^(1+1) * 1.75
E1_8_THRESHOLD = 4.0  # Algorithm 1
E1_16_THRESHOLD = 2.0  # Algorithm 1
E6M2_BIAS = 48
E6M2_NAN_CODE = 0b11111111  # 111111_11
S1P2_MAX = 1.75
S1P2_MIN_POS = 0.25

# S1P2 positive grid (formal model §3)
S1P2_POS_GRID = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75)

# Device tag (eval plan §0)
DEVICE_TAG = "cpu-ref"

# SPDX-FileCopyrightText: Copyright 2025 Martin Lemay <martin.lemay@mines-paris.org>
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

__doc__ = """
Tests functions for CenterlineCollection class - Run with input data
centerline_Collection_test_data*.csv.
"""

import os
import unittest
from typing import Self

import numpy as np
import numpy.typing as npt

from pybend.algorithms.pybend_io import (
    load_centerline_dataset_from_Flumy_csv,
)
from pybend.model.Centerline import Centerline
from pybend.model.Morphometry import Morphometry
from pybend.utils.globalParameters import set_nb_procs

set_nb_procs(1)

# inputs

# output directory for figures
dir_path: str = "tests/data/"
fig_path: str = "tests/.out/"
# create it if absent
if not os.path.exists(fig_path):
    os.makedirs(fig_path)


filepath: str = "tests/data/centerline_flumy2500.csv"
spacing: float = 200  # spacing between channel point (m)
smooth_distance: int = 500  # channel point location smoothing distance (m)
use_fix_nb_points: bool = False
filtering_window: int = 10  # number of points for filtered curvature
sinuo_thres: float = 1.05  # threshold for bends
n = 2  # exponent of curvature distribution function

compute_curvature: bool = True
interpol_props: bool = True
find_bends: bool = True

flow_dir: npt.NDArray[np.float64] = np.array([1.0, 0.0])

nb_procs: int = 3  # number of procs


age, dataset = load_centerline_dataset_from_Flumy_csv(filepath)
centerline = Centerline(
    age,
    dataset,
    spacing,
    smooth_distance,
    use_fix_nb_points,
    filtering_window,
    sinuo_thres,
    n,
    compute_curvature=True,
    interpol_props=True,
    find_bends=False,
)
centerline.find_bends(sinuo_thres, 3)

assert centerline.get_nb_bends() == 46, (
    "Number of bends in invalid. Run tests_Centerline.py first."
)
assert centerline.get_nb_valid_bends() == 34, (
    "Number of bends in invalid. Run tests_Centerline.py first."
)

valid_bend_indexes: list[int] = centerline.get_valid_bend_indexes()

# expected results
#  "Arc_length", "Wavelength", "Sinuosity", "Amplitude", "Extension", "RadiusCurvature", "Skewness", "Roundness", "Wavelength_Leopold", "Amplitude_Leopold"
expected = np.array(
    [
        [
            380.5436,
            790.392,
            1414.5383,
            1649.8798,
            3630.8557,
            4241.879,
            4852.8262,
            2838.8656,
            4886.9827,
            4857.415,
            600.5712,
            1001.7392,
            4669.7962,
            4280.488,
            4440.4856,
            3238.0381,
            1403.1415,
            987.9962,
            3047.2559,
            3456.6151,
            3038.69,
            4864.7946,
            2837.7972,
            3004.599,
            4469.1093,
            1408.3014,
            2849.4233,
            2610.2413,
            5253.7649,
            602.2008,
            1412.4189,
            2248.3439,
            2621.4649,
            603.3264,
            1198.8449,
            4254.9116,
            2817.9652,
            1207.973,
            594.1407,
            2435.0577,
            1210.3271,
            800.2173,
            4424.4804,
            4892.4531,
            5250.1317,
            1797.6553,
        ],
        [
            379.6549,
            788.1816,
            1107.5695,
            1206.8926,
            2434.891,
            2609.1885,
            2945.0158,
            1499.1986,
            2344.8864,
            2342.0798,
            600.3082,
            993.8448,
            2468.0934,
            2189.7348,
            2466.2896,
            2083.0037,
            1369.2976,
            971.9557,
            2338.1839,
            2248.6105,
            1715.1727,
            2460.147,
            1569.4285,
            2793.3688,
            1947.6345,
            1381.7276,
            1563.8871,
            2380.5728,
            2689.7305,
            600.0742,
            1337.2735,
            1519.6748,
            2115.395,
            602.9746,
            1194.9975,
            2790.1382,
            2085.1562,
            1115.9407,
            586.0972,
            1985.6734,
            1035.1926,
            793.8881,
            3071.2782,
            2634.4083,
            2683.9713,
            1591.0283,
        ],
        [
            1.0023,
            1.0028,
            1.2772,
            1.367,
            1.4912,
            1.6257,
            1.6478,
            1.8936,
            2.0841,
            2.074,
            1.0004,
            1.0079,
            1.8921,
            1.9548,
            1.8005,
            1.5545,
            1.0247,
            1.0165,
            1.3033,
            1.5372,
            1.7717,
            1.9774,
            1.8082,
            1.0756,
            2.2946,
            1.0192,
            1.822,
            1.0965,
            1.9533,
            1.0035,
            1.0562,
            1.4795,
            1.2392,
            1.0006,
            1.0032,
            1.525,
            1.3514,
            1.0825,
            1.0137,
            1.2263,
            1.1692,
            1.008,
            1.4406,
            1.8571,
            1.9561,
            1.1299,
        ],
        [
            0.0,
            16.6244,
            412.0354,
            453.8175,
            907.5004,
            762.2989,
            1723.5535,
            927.3957,
            672.9184,
            1023.0159,
            2.689,
            55.4991,
            1048.7565,
            1080.7584,
            1167.5073,
            935.3605,
            136.6252,
            79.6293,
            813.0491,
            781.1501,
            957.6972,
            1288.2501,
            952.8117,
            457.0382,
            1472.9733,
            106.9554,
            989.9583,
            506.2728,
            1992.0483,
            21.8066,
            206.483,
            712.4888,
            636.5879,
            9.4176,
            38.645,
            1138.1956,
            790.524,
            215.2823,
            38.1372,
            571.3892,
            289.7172,
            20.9276,
            1472.2639,
            909.0426,
            1813.3197,
            126.9994,
        ],
        [
            189.8274,
            20.4744,
            443.9391,
            557.0392,
            1315.525,
            1575.9892,
            1760.8234,
            1039.1748,
            1887.9646,
            1997.464,
            100.0244,
            118.0285,
            1746.5138,
            1691.7287,
            1493.6445,
            1253.4306,
            170.8533,
            123.6969,
            991.6862,
            1410.5294,
            1133.0689,
            1660.0612,
            1008.3788,
            700.9652,
            1745.0541,
            325.3689,
            1036.2751,
            516.3366,
            2010.5721,
            103.9038,
            232.5594,
            730.7965,
            647.9582,
            101.9227,
            38.708,
            1367.8156,
            790.5577,
            215.5125,
            105.2858,
            718.019,
            290.4109,
            24.7468,
            1495.0455,
            1881.473,
            1975.882,
            695.4104,
        ],
        [
            1143.4558,
            2884.0155,
            918.764,
            861.2105,
            1570.5965,
            1448.3426,
            1629.3674,
            1143.4473,
            1156.2264,
            995.3332,
            19704.117,
            4001.6243,
            1771.4658,
            1423.6508,
            1894.1779,
            962.431,
            3031.3586,
            4389.5638,
            1368.4067,
            958.4614,
            1119.9732,
            2483.5239,
            1073.9811,
            2683.3148,
            1116.0165,
            2617.64,
            1282.9086,
            2133.9055,
            1550.5441,
            9002.3698,
            2225.4478,
            1095.0679,
            1710.5856,
            6467.1231,
            5737.5446,
            1784.7837,
            1980.9525,
            2019.2421,
            4398.7733,
            1473.8502,
            1701.5313,
            4369.7357,
            1443.2333,
            1373.3324,
            1749.2295,
            1269.5405,
        ],
        [
            -1.0,
            -0.0314,
            0.1839,
            -0.2906,
            -0.4531,
            -0.6212,
            0.0845,
            0.2808,
            -0.5866,
            -0.496,
            0.3334,
            0.2071,
            -0.4856,
            -0.4281,
            -0.3658,
            -0.3659,
            -0.1445,
            -0.191,
            -0.3109,
            -0.5306,
            -0.324,
            -0.4107,
            0.1527,
            -0.3351,
            -0.2679,
            0.4332,
            0.1519,
            0.0725,
            -0.0716,
            -0.337,
            -0.1473,
            0.1107,
            0.0809,
            -0.3363,
            0.0036,
            0.3387,
            0.0035,
            0.0075,
            0.3314,
            -0.3244,
            -0.0193,
            -0.0288,
            -0.0876,
            -0.584,
            -0.228,
            0.8118,
        ],
        [
            2.1024,
            1.6464,
            1.9284,
            1.8211,
            1.8549,
            2.1378,
            2.3753,
            1.7129,
            2.2129,
            2.7163,
            1.3179,
            1.6786,
            2.0595,
            1.9334,
            1.7824,
            2.6147,
            1.7421,
            1.5822,
            2.0807,
            2.4554,
            1.9429,
            1.659,
            1.9714,
            2.4658,
            2.3395,
            2.2675,
            1.6261,
            2.3674,
            2.2886,
            1.6721,
            1.7486,
            2.0144,
            2.0712,
            1.7305,
            1.7918,
            1.8188,
            1.3463,
            1.7318,
            1.8192,
            1.8453,
            1.7556,
            1.6716,
            2.6551,
            2.2123,
            1.9262,
            1.8165,
        ],
        [
            np.nan,
            1979.5398,
            2007.2996,
            1949.2297,
            3535.2485,
            5025.1219,
            6532.7489,
            3582.2045,
            2229.9747,
            4155.2535,
            3944.6233,
            2229.7417,
            3174.9099,
            3927.9753,
            4058.1021,
            4607.3228,
            3850.6171,
            2774.9642,
            3043.3259,
            3836.8153,
            4459.8767,
            3498.7128,
            4229.8596,
            5047.5702,
            1910.9624,
            4878.7549,
            1919.7337,
            5340.5953,
            3087.7047,
            3639.6468,
            2875.667,
            2523.1273,
            2957.8539,
            2331.6328,
            3784.0748,
            3678.407,
            3408.928,
            2796.5447,
            1969.6456,
            2460.1902,
            2951.898,
            3287.3981,
            3905.9438,
            3626.5448,
            5403.8723,
            np.nan,
        ],
        [
            np.nan,
            27.6378,
            549.2705,
            1025.0478,
            1709.2365,
            2975.2273,
            2814.5322,
            1787.6315,
            941.2621,
            2960.3879,
            491.2912,
            37.1052,
            1335.1038,
            3336.9195,
            2862.2325,
            1833.9402,
            151.3545,
            40.4781,
            1376.5082,
            2278.9332,
            2300.7495,
            2938.4018,
            1871.5576,
            961.7334,
            3169.0782,
            659.1385,
            1810.5515,
            1150.2564,
            2593.0941,
            353.3982,
            296.6788,
            1657.2327,
            953.4897,
            49.5221,
            358.2525,
            2031.9986,
            1431.7886,
            285.9486,
            50.8877,
            813.3576,
            344.4016,
            89.9119,
            1942.9146,
            3287.9788,
            3700.6875,
            np.nan,
        ],
    ]
)
sinuo_window_exp = [
    1.3772,
    1.3772,
    1.3772,
    1.3772,
    1.3772,
    1.6695,
    1.7514,
    1.7051,
    2.4213,
    2.402,
    2.5954,
    2.0309,
    2.3653,
    2.4423,
    2.1182,
    1.881,
    1.5936,
    1.5244,
    1.4479,
    1.4508,
    1.6092,
    1.8828,
    2.0908,
    2.453,
    2.5429,
    3.5774,
    2.1652,
    3.0414,
    2.1999,
    2.0643,
    2.3427,
    2.1023,
    2.2855,
    2.7675,
    2.0561,
    1.8384,
    1.6135,
    1.4238,
    1.3879,
    1.4173,
    1.3647,
    1.332,
    1.5488,
    1.9299,
    1.9884,
    1.7941,
]
average_metrics_exp = np.array(
    [
        [
            1573.2419,
            1573.2419,
            1573.2419,
            1573.2419,
            1573.2419,
            3174.2048,
            4241.8536,
            3977.8569,
            4192.8915,
            4194.4211,
            3448.323,
            2153.2418,
            2782.3804,
            3317.3411,
            4463.5899,
            3986.3372,
            3027.2217,
            2517.4153,
            2169.1079,
            2223.7522,
            3180.8537,
            3786.6999,
            3580.4273,
            3569.0636,
            3437.1685,
            2960.6699,
            2908.9447,
            2834.2688,
            3571.1432,
            2927.9828,
            2422.7949,
            2379.1821,
            2427.6387,
            1721.3885,
            1667.995,
            2185.3783,
            2757.2406,
            2760.2833,
            2218.7476,
            2262.0096,
            1653.0927,
            1249.5432,
            1892.8446,
            3372.3836,
            4855.6884,
            3523.8935,
        ],
        [
            1183.4379,
            1183.4379,
            1183.4379,
            1183.4379,
            1183.4379,
            2083.6574,
            2663.0318,
            2351.1343,
            2263.0336,
            2062.0549,
            1762.4248,
            1312.0776,
            1601.0815,
            1883.891,
            2374.7059,
            2246.3427,
            1972.8636,
            1722.6366,
            1690.6102,
            1732.0119,
            2100.6557,
            2141.3101,
            1914.9161,
            2274.3148,
            2103.4773,
            2040.9103,
            1631.0831,
            1818.4555,
            2211.3968,
            1644.9024,
            1542.3594,
            1536.6882,
            1652.4296,
            1393.8295,
            1358.2605,
            1644.636,
            2023.4306,
            1997.0784,
            1644.3331,
            1712.6011,
            1361.612,
            1103.3584,
            1494.4259,
            2166.5249,
            2796.5526,
            2137.4998,
        ],
        [
            1.2281,
            1.2281,
            1.2281,
            1.2281,
            1.2281,
            1.4946,
            1.5882,
            1.7224,
            1.8752,
            2.0172,
            1.7195,
            1.3608,
            1.4936,
            1.6183,
            1.8825,
            1.7699,
            1.4599,
            1.349,
            1.2248,
            1.2204,
            1.5374,
            1.7621,
            1.8524,
            1.6204,
            1.7261,
            1.4631,
            1.7119,
            1.5581,
            1.6239,
            1.4784,
            1.3377,
            1.3731,
            1.3463,
            1.1939,
            1.1806,
            1.2495,
            1.2932,
            1.3196,
            1.2432,
            1.2398,
            1.1686,
            1.0999,
            1.1716,
            1.4352,
            1.7513,
            1.543,
        ],
        [
            357.9955,
            357.9955,
            357.9955,
            357.9955,
            357.9955,
            707.8723,
            1131.1176,
            1137.7494,
            1107.9559,
            874.4433,
            566.2078,
            360.4013,
            532.4901,
            728.338,
            1099.0074,
            1061.2087,
            746.4977,
            579.7806,
            491.166,
            452.6134,
            850.6321,
            1009.0325,
            1066.253,
            899.3667,
            960.9411,
            678.989,
            856.629,
            769.04,
            1162.7598,
            1006.9274,
            740.1126,
            733.2067,
            713.8829,
            391.2443,
            349.2848,
            507.067,
            655.7882,
            714.6673,
            545.5348,
            550.7057,
            381.01,
            227.0907,
            478.487,
            800.7447,
            1398.2087,
            970.1596,
        ],
        [
            505.361,
            505.361,
            505.361,
            505.361,
            505.361,
            1149.5178,
            1550.7792,
            1458.6625,
            1562.6543,
            1641.5345,
            1328.4843,
            738.5056,
            990.5077,
            1185.4237,
            1643.9623,
            1479.6013,
            972.6428,
            760.4063,
            634.9168,
            674.1914,
            1178.4282,
            1401.2198,
            1267.1696,
            1123.1351,
            1151.466,
            923.7961,
            1035.566,
            905.7587,
            1187.7279,
            1057.238,
            782.3451,
            769.458,
            745.158,
            428.3092,
            379.8464,
            577.4402,
            732.3604,
            791.2953,
            619.7929,
            639.4381,
            423.9572,
            270.795,
            526.7016,
            1133.7551,
            1784.1335,
            1335.6462,
        ],
        [
            1475.6085,
            1475.6085,
            1475.6085,
            1475.6085,
            1475.6085,
            1293.3832,
            1549.4355,
            1407.0524,
            1309.6804,
            1098.3356,
            7285.2255,
            8233.6915,
            6618.1351,
            2398.9136,
            1696.4315,
            1426.7532,
            1962.6558,
            2569.3828,
            2437.94,
            2436.9476,
            1148.9471,
            1520.6528,
            1559.1594,
            2080.2733,
            1624.4375,
            2138.9904,
            1672.1884,
            1787.6176,
            1655.7861,
            5276.457,
            4259.4539,
            3468.3574,
            3116.803,
            2874.5561,
            3752.5803,
            3359.021,
            3167.7603,
            1928.3261,
            2545.9379,
            2331.5204,
            2314.8699,
            2792.6265,
            2677.4248,
            2395.4338,
            1521.9317,
            1509.385,
        ],
        [
            -0.3182,
            -0.3182,
            -0.3182,
            -0.3182,
            -0.3182,
            -0.455,
            -0.3299,
            -0.0853,
            -0.0738,
            -0.2673,
            -0.2497,
            0.0148,
            -0.1103,
            -0.2355,
            -0.4265,
            -0.3866,
            -0.2921,
            -0.2668,
            -0.2531,
            -0.2942,
            -0.3885,
            -0.4218,
            -0.194,
            -0.1977,
            -0.1501,
            -0.0566,
            0.1057,
            0.0974,
            0.0509,
            -0.2043,
            -0.1853,
            -0.1113,
            -0.0729,
            -0.073,
            -0.0353,
            0.0395,
            0.1153,
            0.1166,
            0.1703,
            0.0713,
            -0.0003,
            -0.0067,
            -0.0257,
            -0.2335,
            -0.2999,
            0.2919,
        ],
        [
            1.8706,
            1.8706,
            1.8706,
            1.8706,
            1.8706,
            1.9379,
            2.1227,
            2.0753,
            2.1004,
            2.214,
            2.0824,
            1.9043,
            1.9431,
            1.8905,
            1.9251,
            2.1102,
            2.0464,
            1.9304,
            2.0049,
            1.9651,
            2.1597,
            2.0191,
            1.8578,
            2.0321,
            2.2589,
            2.3576,
            2.0777,
            2.1501,
            2.094,
            1.9804,
            1.9031,
            1.9309,
            1.959,
            1.8912,
            1.902,
            1.8853,
            1.6523,
            1.6323,
            1.679,
            1.7123,
            1.6996,
            1.7647,
            1.9494,
            2.1797,
            2.2645,
            1.8714,
        ],
        [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            3503.2,
            5031.0398,
            5046.6918,
            4114.976,
            3322.4776,
            3443.2838,
            3443.2062,
            3376.1321,
            3110.8756,
            3720.3291,
            4197.8001,
            4172.014,
            3822.7516,
            3569.0575,
            3376.4306,
            3780.006,
            3931.8016,
            4062.8164,
            4258.7142,
            3729.4641,
            3945.7625,
            2903.1503,
            3512.5116,
            3449.3446,
            3363.6758,
            3201.0062,
            3031.5364,
            3016.7999,
            2672.0702,
            2899.1722,
            3055.0192,
            3623.8033,
            3294.6266,
            2963.3813,
            2862.7431,
            2717.4413,
            2693.1353,
            2915.0151,
            3606.6289,
            4312.1203,
            np.nan,
        ],
        [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            1903.1705,
            2499.6653,
            2525.797,
            1847.8086,
            1896.4272,
            1464.3137,
            1162.9281,
            1205.972,
            1569.7095,
            2511.4186,
            2677.6974,
            1615.8424,
            1222.0013,
            850.5702,
            961.8185,
            1985.397,
            2506.0282,
            2370.2363,
            1923.8976,
            2000.7897,
            1596.65,
            1879.5894,
            1697.2562,
            1851.3007,
            1473.2462,
            1081.057,
            1225.101,
            1170.7787,
            739.2308,
            754.6243,
            1010.0991,
            1274.0132,
            1249.9119,
            950.1559,
            922.7962,
            585.2768,
            316.9015,
            648.2947,
            1773.6018,
            2977.1936,
            np.nan,
        ],
    ]
)


class TestsMorphometry(unittest.TestCase):
    def test_initialization(self: Self) -> None:
        """Test of Morphometry initialization."""
        morph: Morphometry = Morphometry(centerline)
        self.assertIsNotNone(morph.centerline)
        self.assertEqual(age, morph.centerline.age)

    def test_compute_bend_arc_length(self: Self) -> None:
        """Test of compute_bend_arc_length method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_arc_length(i)
            for i in range(morph.centerline.get_nb_bends())
        ]
        print(obs)
        self.assertSequenceEqual(
            obs, expected[0].tolist(), "Arc length are not equal."
        )

    def test_compute_bend_wavelength(self: Self) -> None:
        """Test of compute_bend_wavelength method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_wavelength(i)
            for i in range(morph.centerline.get_nb_bends())
        ]
        print(obs)
        self.assertSequenceEqual(
            obs, expected[1].tolist(), "Wavelength are not equal."
        )

    def test_compute_bend_sinuosity(self: Self) -> None:
        """Test of compute_bend_sinuosity method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_sinuosity(i)
            for i in range(morph.centerline.get_nb_bends())
        ]
        print(obs)
        self.assertSequenceEqual(
            obs, expected[2].tolist(), "Sinuosity are not equal."
        )

    def test_compute_bend_amplitude(self: Self) -> None:
        """Test of compute_bend_amplitude method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_amplitude(i)
            for i in range(morph.centerline.get_nb_bends())
        ]
        print(obs)
        self.assertSequenceEqual(
            obs, expected[3].tolist(), "Amplitude are not equal."
        )

    def test_compute_bend_extension(self: Self) -> None:
        """Test of compute_bend_extension method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_extension(i)
            for i in range(morph.centerline.get_nb_bends())
        ]
        print(obs)
        self.assertSequenceEqual(
            obs, expected[4].tolist(), "Extension are not equal."
        )

    def test_compute_bend_radius(self: Self) -> None:
        """Test of compute_bend_radius method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_radius(i)
            for i in range(morph.centerline.get_nb_bends())
        ]
        print(obs)
        self.assertSequenceEqual(
            obs, expected[5].tolist(), "Radius are not equal."
        )

    def test_compute_bend_asymmetry(self: Self) -> None:
        """Test of compute_bend_asymmetry method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_asymmetry(i)
            for i in range(morph.centerline.get_nb_bends())
        ]
        print(obs)
        self.assertSequenceEqual(
            obs, expected[6].tolist(), "Asymmetry are not equal."
        )

    def test_compute_bend_roundness(self: Self) -> None:
        """Test of compute_bend_roundness method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_roundness(i)
            for i in range(morph.centerline.get_nb_bends())
        ]
        print(obs)
        self.assertSequenceEqual(
            obs, expected[7].tolist(), "Roudness are not equal."
        )

    def test_compute_bend_wavelength_leopold(self: Self) -> None:
        """Test of compute_bend_wavelength_leopold method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_wavelength_leopold(i)
            for i in range(1, morph.centerline.get_nb_bends() - 1, 1)
        ]
        self.assertSequenceEqual(
            obs,
            expected[8].tolist()[1:-1],
            "Leopold wavelength are not equal.",
        )

    def test_compute_bend_amplitude_leopold(self: Self) -> None:
        """Test of compute_bend_amplitude_leopold method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_amplitude_leopold(i)
            for i in range(1, morph.centerline.get_nb_bends() - 1, 1)
        ]
        self.assertSequenceEqual(
            obs, expected[9].tolist()[1:-1], "Leopold amplitude are not equal."
        )

    def test_compute_bends_morphometry_all(self: Self) -> None:
        """Test of compute_bends_morphometry method."""
        morph: Morphometry = Morphometry(centerline)
        obs = morph.compute_bends_morphometry(valid_bends=False).to_numpy()
        self.assertTrue(np.array_equal(obs, expected.T, True))

    def test_compute_bends_morphometry_valid_bends(self: Self) -> None:
        """Test of compute_bends_morphometry method."""
        morph: Morphometry = Morphometry(centerline)
        obs = morph.compute_bends_morphometry(valid_bends=True).to_numpy()
        self.assertTrue(
            np.array_equal(obs, expected.T[valid_bend_indexes], True)
        )

    def test_compute_bend_sinuosity_moving_window(self: Self) -> None:
        """Test of compute_bend_sinuosity_moving_window method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_bend_sinuosity_moving_window(i, 5000.0)
            for i in range(morph.centerline.get_nb_bends())
        ]
        print(obs)
        self.assertSequenceEqual(
            obs,
            sinuo_window_exp,
            "Sinuosity over moving window are not equal.",
        )

    def test_compute_average_metric_window(self: Self) -> None:
        """Test of compute_average_metric_window method."""
        morph: Morphometry = Morphometry(centerline)
        obs = [
            morph.compute_average_metric_window(i, 5000.0).to_numpy()
            for i in range(morph.centerline.get_nb_bends())
        ]
        self.assertTrue(np.array_equal(obs, average_metrics_exp.T, True))

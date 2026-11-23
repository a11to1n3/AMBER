"""Bulk 1-D output hashes recorded on NumPy at 10ea82c."""

from hashlib import sha256

import numpy as np
import pytest

from ambr.gpu import EVT_INIT, GPU_AVAILABLE, make_keyed_device_rng, to_host

ROUNDING_SEED = 5042506492838628837
SIZES = (1, 7, 1000, 100000)
DTYPES = (np.float64, np.float32, np.int64)

# Each row holds float64, float32 and int64 hashes, in that order.
GOLDEN = {
    (1, 1, False): (
        "839a47d0a903136f4c9e6e02fad1445c91a457d412985395cd25813cd32d8f2f",
        "b489d892da1e5560d6f05801d95903ac281baa688d139e27440815e05cd0305e",
        "ef565b619f052317c46b5be4ab545bef4a59846c65e66d3abc98f1390e7facda",
    ),
    (1, 1, True): (
        "1ecfde7b4e51ff78c7dbbe852c5debf4e5fee1d13952ea21bd7e70678054fdd8",
        "8af20257e83590dcfee56579223a6ce2b7aa086d8293289056d510e33e297136",
        "b3c764f7d2129d66da427040928416d21ba34a784ca5896f916226fd1b794fe5",
    ),
    (1, 7, False): (
        "b556124ab1daab263cbef51f84ee4f3a1950386414c9a66fef45cc8190e3317d",
        "f3abeca02d292fbd48836b47abe46b77d81406d0be3f4ffb73b17bf26ca19649",
        "f85c9c1376f5086838d3803e59e60d38c683de1c4c70f798b7069f88fc0b95a9",
    ),
    (1, 7, True): (
        "c251afbaf6a6d148856df236885eff6e2b46ba07e2abbe8c3abe136642e61d36",
        "6ecccb725bd08b3747cbadfdb722c93ef9f6413f0eb70bd39003bc08d498afb3",
        "1b67e5e127cf71c3d43f5549706ce0b7d9e1b8e438c043cb8fea720ec440698b",
    ),
    (1, 1000, False): (
        "751aa195880e8e1bf2c18411865075dc55e2ffd3bcf50133d9bc2fad412e48c8",
        "c400bfc33b56e67e5febc2ba9d64038a8b39ca0cde939d854e0dfdb6f687d17e",
        "b47d140ad384912712b4f511eb1471169e564a0792ee3967bda878ac5514031d",
    ),
    (1, 1000, True): (
        "4071e5b28aa0434fdeafe053882d13a50a997193e5d0780f7141464da1ed941f",
        "b012363b66fb58d0c1176906c3ff927b5ae6d79b70f7c3ab027bf14b8ef614a9",
        "cca003e9af442642f65829894984898cc8a585088abd5eab7d07dbbcb46a10d1",
    ),
    (1, 100000, False): (
        "c45e41f6a1b85ce6113f507bd654d94dc50616e05e61503615344749b21fbfc5",
        "8b3acc2753085736876e1d1969bb28cfd26f8ff216d63b5852324ddb16d86aa9",
        "041df4561c95d9a9dcd1fec25fbb11f8da6cd21c50a06d9f82689ac84835c3bd",
    ),
    (1, 100000, True): (
        "09fa2b42ca67fc288ee28903f58386e52c1afa52ee55db8d5fccddf5f5bcd778",
        "4ca7f3875f0b3812fe78bd392c8c7e0082b64b811fd12808c637afaa56832f7e",
        "2c44fc6e66a266d63ab29282efac6530822191a7cd51be0b8d0e296d3ea0b3e0",
    ),
    (42, 1, False): (
        "dfb49299e93501c439446b3cf2139addff18636efd2e04529644f96a4a9359f0",
        "4b2cf3c521eec6ed87abb912b63594b667c8c9561451bc8e51b5c0ce82b7c253",
        "4fec8bec80a3f974a0c21e782e3896e259706dc1195b5773ded9515fd9b8086b",
    ),
    (42, 1, True): (
        "0a94d17b31c1a32bb14900156120570427fa2bfbbe0ba869e61aabfc97451221",
        "8a52bec040e1353f4980fa18456f56e9dc634af38e4257f25555710d9b0e9584",
        "084c2e9814dc75271b09ed53975d6c8e79e621ff247bd6035de4dcf938dc1033",
    ),
    (42, 7, False): (
        "bfcaff605257085b3ffede1ef710f2fa953b1335cf256c71f59cc4714a1fef54",
        "a2c6789b9730cf43be3b8e6c68aaa2805cefb28acfe62420c2915731579617ac",
        "ee6a3228421081c9c1f7d6e6c49a83027248128834d3c11b2e0bd7f35b2891de",
    ),
    (42, 7, True): (
        "00a6be11e015bc66511c0ef8122c4605a21c54e8d73c8dd5ca1bdf9234c9ae2d",
        "8de2d7fda1874c81cacac9afdfafcf6075a9207261bf65f876c30f5b430a661e",
        "6c6b38b35d8a969f7f0b584243047d6fbcc02c9a2891c0dbada5ac4ce03fe53e",
    ),
    (42, 1000, False): (
        "b43666cb77f9b057222ed355566ff6136bdf0ac366b7939eb521d4b038c8e46b",
        "6adb231f739eac6b4d80a986b05217ffe94b1bbf4ac0baf1281277444b86a098",
        "b480891a5fe1968184f1f189ac561a1cdee72112dc5d15ed31f4f953b636998e",
    ),
    (42, 1000, True): (
        "225586d2aaebcdcb7f39b6eae35730e29e9bf66020c79f66ebfe0af8e17cc5ce",
        "881ba8710a5e34cce589d27c41b6160d0d1e9c1f20335ac2504f2c401314e813",
        "1062d7661baa3956aed47015cd82da0c207819efd4368544bee7eae14712d760",
    ),
    (42, 100000, False): (
        "23f49a4fcd299414b9f436afa5a92d674d1b7493aa677c637f6de8efabce5771",
        "fefaa6b18131a3afae054417ac42e042b44cbaa633988c7b2f608da9a9f8ce99",
        "8a404151632dfd232efcef72658dc1e3bdde53a04bef5f0d20b15591cfb1eec1",
    ),
    (42, 100000, True): (
        "e74fa0100a1913ceef9acc4493d8adc24267f847c21296d6277ca34a64515d1f",
        "bc24c976cf9d34d2749d332fe5091f34cfe83b0c534fd523642d567fa0387e31",
        "0d9a056cd2cdcb9116c61c9e3bab86d70f6da9542a4976df457e27cb1d5d0e4e",
    ),
    (5042506492838628837, 1, False): (
        "f39400851e7c544abb15add91558fa19af2a3e0dfe0acc0c8633a7803d78d28d",
        "295efc2f87595eb3e2a660e8b5eac8fb6f518854c42464fd2269fbd5fb70f8b5",
        "c96d15cc10fbcce51e6834a5071293319c47a9d79a0a533bf89b95cad0cb53b0",
    ),
    (5042506492838628837, 1, True): (
        "e635c5cac15ba2c32a6c9c9dfac69aaa0fbe00a2cfc4d1a8c850e18217b18b8e",
        "dd6624e4d5576a03a9364e0d33b026dfcaa3d8ed35a44434dc25b57889607a48",
        "63fc7fdfe2c1580386ccf40e325e1a032c83b0725964f0e1b23620ff4e8f8a71",
    ),
    (5042506492838628837, 7, False): (
        "39b4c65f0e4f9af6ebf349839b0646dd6d057adfeaeb9b9d74a659443809ec87",
        "b09280bd94a63a29336b0aad16124e8ebdf0ecb61b30edb5ed6b622328cc916b",
        "b72d3ee82601e37ca2ae39e70d5d52054f0d0986a550592c0c676e0c2895dc2d",
    ),
    (5042506492838628837, 7, True): (
        "27c176c9f95457717d4f49098c47a2892c8a80aed7b65ea6e64357c309047e9b",
        "285918430832b75e5d664deff8674e592678f0d5944535b25dd6dd1bb94d0e9a",
        "cae36bf687da95b3682fdad8cf5dbf77e2b106c15c0b7ef523d599c06da2fc52",
    ),
    (5042506492838628837, 1000, False): (
        "a6f34b367d6dfcb41d9019201e6770b1b43157182c9217e1c2c02a9fbadc8761",
        "43d09b9e127274fc12419fbacf29664049e06526b481d4281ddd45f6e23708c0",
        "83c4d3259b482629b69ffa0d2e4e765db195150a4f24ce3003d13b0e0de3b07b",
    ),
    (5042506492838628837, 1000, True): (
        "160cc08aecbc18737d6bae8a8b95a4f425b1dd10f43f463d70cd8ba334eeb2af",
        "9252a1ff9b122e71b46fcee6ef64dbe5121c34f2bba723f33eb27dc0da3432c0",
        "70c76654ee7c77b6ffad18431416b3891a5f2141f8cc8a99e26245d8d919fa2d",
    ),
    (5042506492838628837, 100000, False): (
        "2c931a37e7fe7bd29070084fdace60996f08992f1694787961f557b544604b8e",
        "b6134be100b854d63c9344ef5c36a80559a0a5c9276340a94227e3888e8abcc3",
        "bcfe1c1a18197b97abe1c928261e7a943d8eba3100ec957e6fe23712419ee1e0",
    ),
    (5042506492838628837, 100000, True): (
        "0cd76032e0f31355939f4bb7712e198984351525e8935f68111b4b2830bb2d98",
        "e62f9b12e00e262768a4957fd981751f120f29683538f5e6d0eeb48b8a4d5007",
        "60240e7841681dcc527f6a08ac5c59c83cf5284e86c1c131a92d0875696eed8a",
    ),
}


def bulk_sequence(seed, size, keyed, dtype, xp):
    rng = make_keyed_device_rng(seed, xp=xp)
    for step, event in ((0, EVT_INIT), (1, 1), (2, 3), (7, 1)):
        rng.begin_step(step, event)
        if keyed:
            rng.set_agent_keys(3 * xp.arange(size, dtype=xp.int64) - 17)
        if dtype is np.int64:
            yield False, rng.integers(-1009, 130_003, size=size, dtype=dtype)
            yield False, rng.integers(13, size=size, dtype=dtype)
            yield False, rng.integers(3, 1001, size=size, dtype=dtype, endpoint=True)
        else:
            yield True, rng.random(size, dtype=dtype)
            yield False, rng.uniform(-2.0, 3.0, size, dtype=dtype)
            yield True, rng.uniform(size=size, dtype=dtype)


@pytest.mark.parametrize("seed", [1, 42, ROUNDING_SEED])
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("keyed", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_bulk_only_output_matches_base(seed, size, keyed, dtype):
    draws = [out for _, out in bulk_sequence(seed, size, keyed, dtype, np)]
    assert sha256(np.concatenate(draws).tobytes()).hexdigest() == GOLDEN[seed, size, keyed][
        DTYPES.index(dtype)
    ]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="requires CuPy and a CUDA device")
@pytest.mark.parametrize("seed", [1, 42, ROUNDING_SEED])
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("keyed", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_bulk_golden_sequences_match_cuda(seed, size, keyed, dtype):
    import cupy as cp

    host = bulk_sequence(seed, size, keyed, dtype, np)
    device = bulk_sequence(seed, size, keyed, dtype, cp)
    for (_, reference), (_, actual) in zip(host, device, strict=True):
        assert actual.dtype == reference.dtype
        assert to_host(actual).tobytes() == reference.tobytes()

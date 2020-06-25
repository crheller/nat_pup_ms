"""
Determine fastest way to load saved decoding results
"""
import timeit
import charlieTools.nat_sounds_ms.decoding as decoding
import json
import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
jsonpickle_pd.register_handlers()
import pickle

local = False
if local:
    fpkl = '/home/charlie/Desktop/test_files/test.pickle'
    fjson = '/home/charlie/Desktop/test_files/test.json'

else:
    fpkl = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/bbl099g/dprime_jk1_eev_zscore_nclvz_fixtdr2_TDR.pickle'
    fjson = '/auto/users/hellerc/results/nat_pupil_ms/dprime_new/bbl099g/dprime_jk1_eev_zscore_nclvz_fixtdr2_TDR.json'

loader = decoding.DecodingResults()

# standard pickle method
start = timeit.default_timer()
res = loader.load_results(fpkl)
end = timeit.default_timer()

print("Time (pickle) = {0} s".format(end - start))

# json serialized
start = timeit.default_timer()
f = open(fjson, 'r')
froz = json.load(f)
#res = jsonpickle.decode(froz)
end = timeit.default_timer()

print("Time (json serialized string) = {0} s".format(end - start))

# json serialized & decode the string
start = timeit.default_timer()
loader.load_json(fjson)
end = timeit.default_timer()

print("Time (json serialized string + object decoding) = {0} s".format(end - start))


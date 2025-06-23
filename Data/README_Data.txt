Information about the files in the ./Data directory, and files within.

MS_mass_age.csv -- main-sequence lifetime - mass used from MIST isochrones

multidim.py -- COSMIC multidim.py function with updated IMF (Kroupa) and interpolated binary fraction to Winters+2019 for low-mass stars (instead of f_binary = 0 at M=0.08)

mutlidim_binaries_initial.parquet -- 5 x 10^5 sampled binaries following Moe and Di Stefano (2017), sampled using multidim.py in COSMIC

singles_imf.npy --  corresponding masses and number of single star population from the above binary sampling (preserved multiplicity statistics and IMF)

triples_catalog.csv -- observed catalog of resolved triples (R<0.1)

triples_all_r_chance_score.csv -- entire resolved triples catalog (all R, note many will be false matches)

resolved_quads_500pc.csv - catalog of resolved quadruples

unresolved_higher_order_multiples.csv -- catalog of triples where one component is an unresolved subsystem (multiplicity 4+)






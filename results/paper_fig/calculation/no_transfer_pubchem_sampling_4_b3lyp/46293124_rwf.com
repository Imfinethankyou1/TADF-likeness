%chk=calculation/no_transfer_pubchem_sampling_4_b3lyp/46293124_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_4_b3lyp/46293124_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 6.9297723541       -1.9964547680       -0.4362596455
C                 5.5222247261       -2.5141023523       -0.7446536595
N                 4.4800549993       -1.7542060117       -0.0718802893
N                 4.2228330672       -2.0197386325        1.2351743837
C                 3.3090431492       -1.1223914883        1.5987922866
C                 2.7511708224       -1.1150135327        2.9891543561
C                 2.9745884980       -0.2586561073        0.5213114607
C                 1.9707872204        0.8598775700        0.5371298269
N                 0.6155550596        0.3261914501        0.6503656692
C                -0.3587049609        0.8546515530        0.0122251139
N                -1.5538584624        0.0873570063        0.0130404387
C                -2.7262625608        0.2881761570       -0.6790738671
O                -2.8819945855        1.2054452169       -1.4894597605
C                -3.8137557814       -0.7068125799       -0.4095754523
C                -4.8127493842       -0.8393817056       -1.3785656348
C                -5.8631748108       -1.7460748154       -1.2059314197
C                -6.9181455300       -1.9168346129       -2.2744761516
C                -5.9003416789       -2.5039191645       -0.0304385552
C                -4.9237118485       -2.3784347594        0.9671702487
C                -5.0152220374       -3.1924743045        2.2373573551
C                -3.8789734321       -1.4742959611        0.7629949749
N                -0.4241060993        2.0340367003       -0.7136298026
C                 0.0710786370        3.2779523406       -0.2838711582
C                 0.4339226233        3.5834884159        1.0351847674
C                 0.8932930858        4.8593995450        1.3680145269
C                 0.9779320973        5.8606512934        0.4008306778
C                 0.5946402062        5.5826649719       -0.9142014561
C                 0.1541439946        4.3091914858       -1.2313949436
F                -0.2331760850        4.0312677368       -2.4987603226
C                 3.7527201976       -0.6961529151       -0.5417574309
C                 3.8509211091       -0.2062485626       -1.9533218770
H                 7.1089208874       -2.0201897365        0.6427062631
H                 7.0582385786       -0.9654279176       -0.7839035848
H                 7.6823909554       -2.6219665208       -0.9299322029
H                 5.3225334440       -2.4845335812       -1.8203175460
H                 5.4181217100       -3.5517916824       -0.4150889696
H                 1.6582292006       -1.1920808483        2.9714989191
H                 3.1596566958       -1.9530917400        3.5605642283
H                 3.0007904275       -0.1861336495        3.5194548454
H                 2.1743367065        1.5200809303        1.3935184472
H                 2.0915858890        1.4865061459       -0.3575267215
H                -1.4182511875       -0.7937597581        0.4916014783
H                -4.7540392866       -0.2206091742       -2.2686069157
H                -6.5944456488       -2.6355475662       -3.0389087153
H                -7.1241577191       -0.9709029980       -2.7860585992
H                -7.8583098727       -2.2886012433       -1.8541467427
H                -6.7183620591       -3.2069411186        0.1189966111
H                -5.0582297288       -4.2666164976        2.0203708871
H                -4.1532151726       -3.0175870452        2.8884685080
H                -5.9190088638       -2.9428180474        2.8069266776
H                -3.1405479831       -1.3443564132        1.5503165681
H                -1.2231020828        2.0751353853       -1.3499586302
H                 0.3384613638        2.8184285977        1.7980477315
H                 1.1726981333        5.0701463168        2.3962789382
H                 1.3315205911        6.8532306383        0.6624435173
H                 0.6336734901        6.3335184560       -1.6967816595
H                 3.1553149609        0.6215893364       -2.1134745437
H                 3.5974389769       -0.9907393333       -2.6778653011
H                 4.8572530038        0.1559038948       -2.2001151486



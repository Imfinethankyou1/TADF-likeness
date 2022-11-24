%chk=calculation/no_transfer_pubchem_sampling_4_b3lyp/129363522_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_4_b3lyp/129363522_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -5.6542984217       -1.9106305298       -0.2299112689
O                -4.4304621356       -1.5989021115        0.4149003402
C                -3.7468984955       -0.4902066548        0.0034944120
C                -4.1645449613        0.3672867235       -1.0204675873
C                -3.3733917058        1.4740805969       -1.3430955282
C                -2.1856065439        1.7297491008       -0.6650956749
C                -1.7619666402        0.8712390742        0.3626115820
C                -0.4511671925        1.1132657648        1.0820634732
C                 0.7360902898        0.4452391641        0.3629793358
C                 2.1099127452        0.6998696968        1.0479947232
C                 2.1022186914        0.1808870364        2.4255954215
N                 2.0717210500       -0.2420684667        3.5066805758
C                 3.2628961819        0.1227306375        0.2290239864
C                 3.9061102112        0.9328521382       -0.7139065862
C                 4.9212227346        0.4163395615       -1.5196461326
C                 5.3069967269       -0.9185669271       -1.3881581519
C                 4.6727778627       -1.7311768747       -0.4468947608
C                 3.6561325508       -1.2144779035        0.3569347735
C                -2.5500609801       -0.2322373426        0.6873601757
H                -6.0233382299       -2.8184620929        0.2509597707
H                -6.3936695583       -1.1083220912       -0.1044782819
H                -5.5100462273       -2.1018466638       -1.3017563770
H                -5.0878630776        0.1902487146       -1.5598095171
H                -3.6999897197        2.1442511999       -2.1342019913
H                -1.5863829958        2.5992393351       -0.9245708174
H                -0.5125841285        0.7328374169        2.1083964599
H                -0.2640520598        2.1931446951        1.1553087952
H                 0.8102584324        0.8217879239       -0.6637262275
H                 0.5703684540       -0.6358169248        0.2904161623
H                 2.2512129132        1.7868676717        1.1231433005
H                 3.6125934911        1.9754223150       -0.8173753832
H                 5.4138846587        1.0589102260       -2.2443823291
H                 6.1006473797       -1.3214711223       -2.0113240409
H                 4.9716545316       -2.7696538741       -0.3331415019
H                 3.1767542806       -1.8490857134        1.0977409231
H                -2.2571291125       -0.9133246993        1.4818499470



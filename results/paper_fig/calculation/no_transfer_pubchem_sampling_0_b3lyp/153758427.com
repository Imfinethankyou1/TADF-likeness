%chk=./calculation/no_transfer_pubchem_sampling_0_b3lyp/153758427.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -3.3080287668        1.2355280650       -2.6053218364
C                -2.9076813189        0.8946227845       -1.1750717027
C                -3.7224885673        1.6974270849       -0.1636393016
C                -3.3960984165        1.3169864832        1.2885715563
C                -1.9563081388        1.6841044661        1.5820310353
O                -1.6062035536        2.8145130432        1.8037314049
C                -0.9900872951        0.5374279433        1.5466746131
C                -1.1503312417       -0.5705463522        2.3751201467
C                -0.2079189315       -1.5811757751        2.3167401666
C                 0.8581888651       -1.4872860368        1.4231729028
O                 1.8142764664       -2.4287260491        1.2909786724
C                 1.7678771123       -3.5804115800        2.1048659543
C                 0.9341939452       -0.3257617940        0.6097727341
C                 1.9597017750       -0.0522549702       -0.3793820810
C                 2.9954601412       -0.8231800955       -0.7305262185
C                 3.9788159031       -0.4331937994       -1.7538765643
O                 4.9261566296       -1.1152813859       -2.0679367022
O                 3.7227511349        0.7613785533       -2.3125871097
C                 4.6430616132        1.1896332392       -3.3028140408
N                 0.0273125383        0.6389678886        0.7096272422
H                -2.7185923035        0.6589906804       -3.3139137378
H                -3.1444571400        2.2922749682       -2.8028758046
H                -4.3603639976        1.0145651787       -2.7731030411
H                -3.0616575730       -0.1736671039       -0.9997111491
H                -1.8456351027        1.1048470756       -1.0361280956
H                -4.7877427463        1.5274727528       -0.3384570709
H                -3.5234389425        2.7619180081       -0.2994555634
H                -4.0440349367        1.8901608670        1.9544253457
H                -3.5665794033        0.2505873058        1.4370959965
H                -1.9738677463       -0.6325509306        3.0696726708
H                -0.3104896913       -2.4369515940        2.9659900516
H                 0.8607841691       -4.1644820286        1.9229622983
H                 2.6379373238       -4.1701123858        1.8213796454
H                 1.8355181574       -3.3247285921        3.1663937195
H                 1.8246779677        0.9053802834       -0.8643867044
H                 3.1853598330       -1.7854952913       -0.2844682844
H                 4.2808695725        2.1503006631       -3.6597918382
H                 4.6864137166        0.4667630458       -4.1205633597
H                 5.6426499488        1.2919543844       -2.8744849502



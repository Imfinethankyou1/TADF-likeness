%chk=calculation/no_transfer_pubchem_sampling_0_b3lyp/10332962_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_0_b3lyp/10332962_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 5.0890703949       -2.1622938021       -0.4218343253
C                 4.6615236881       -0.9857072113       -1.2878018354
O                 3.2244542390       -0.8816796751       -1.3712862833
C                 2.6075122390       -0.2098434338       -0.3549149388
O                 3.2243005917        0.3097042370        0.5524115317
C                 1.1290026436       -0.2247837080       -0.5234108466
C                 0.5496573350       -1.1343007989       -1.4374739043
N                -0.7552567563       -1.2858489889       -1.6674794234
C                -1.5068793653       -0.4676855380       -0.9324732623
C                -1.0966969508        0.4950153946        0.0124741561
N                -2.2091934278        1.1326213968        0.5488773206
C                -3.2442551522        0.5816947095       -0.0449987840
C                -4.6681170442        0.9474004781        0.2143124365
N                -2.8858648061       -0.3951310577       -0.9562237208
C                -3.7500645758       -1.2106304105       -1.7921642458
C                 0.2899562973        0.6391897501        0.2388563423
O                 0.8689730525        1.5179761046        1.0589461536
C                 0.1031530600        2.4016911173        1.9101666869
C                -0.3069826359        1.7138769601        3.2028720403
H                 4.7741505493       -2.0097042934        0.6141716572
H                 4.6534129698       -3.0968310001       -0.7909874238
H                 6.1808483754       -2.2612929689       -0.4387826389
H                 4.9830390731       -1.1162408497       -2.3245993680
H                 5.0619017901       -0.0460911606       -0.8996203925
H                 1.2121263629       -1.7776409762       -2.0055869654
H                -5.2474838354        0.0905084439        0.5805986096
H                -5.1645940301        1.3155247923       -0.6924010184
H                -4.6941372137        1.7344993420        0.9698512387
H                -3.1069376619       -1.8658159858       -2.3820985662
H                -4.4239379980       -1.8203031123       -1.1812328477
H                -4.3448733858       -0.5856344304       -2.4663156091
H                -0.7647665387        2.7829226763        1.3704320268
H                 0.7994281561        3.2207621916        2.1081008500
H                -0.7722160766        2.4451606125        3.8744198429
H                -1.0363303778        0.9225896492        3.0091079152
H                 0.5660790146        1.2863215461        3.7060915912



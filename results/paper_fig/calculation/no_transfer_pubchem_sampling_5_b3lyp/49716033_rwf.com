%chk=calculation/no_transfer_pubchem_sampling_5_b3lyp/49716033_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_5_b3lyp/49716033_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -4.1095042274       -1.1122706128       -1.2927409387
N                -4.0017702267       -0.9250381361        0.1430763958
C                -4.8925562305       -0.1869112963        0.8874036145
C                -4.4654473866       -0.1674220469        2.1977920738
C                -3.2662356462       -0.9235701755        2.2550780122
C                -2.9891723832       -1.3805185733        0.9771959844
C                -1.8250061002       -2.2056627445        0.5138089849
N                -0.5718829201       -1.4595239171        0.2274805665
C                -0.4182298605       -0.8342830130       -1.0904931167
C                -0.6682475908        0.6576719219       -1.1140455063
C                -1.3535176101        1.4462878673       -0.2277620966
C                -1.3966574218        2.8222820575       -0.6160084724
C                -0.7481797924        3.0665158198       -1.7921197138
S                -0.0590423984        1.6130476228       -2.4496501169
C                 0.2430956862       -1.2246235047        1.3056964846
O                -0.0521287622       -1.5833946876        2.4433044373
C                 1.5614803547       -0.5046240424        1.1245465918
C                 2.0467387043        0.3027589591        2.0930435341
C                 3.3354855369        0.9797591108        1.9564891287
O                 3.7941879065        1.7404160082        2.8057615574
C                 4.0481784140        0.6575279700        0.6973729576
C                 5.3102718193        1.2029306091        0.4111762164
C                 5.9666377706        0.8816817565       -0.7672889168
C                 5.3680596517        0.0018774435       -1.6846969055
C                 4.1207952508       -0.5513702061       -1.4255889552
C                 3.4705417602       -0.2140696545       -0.2348502138
O                 2.2360797915       -0.7844994837       -0.0241511587
H                -3.9255399343       -2.1559670502       -1.5660964854
H                -3.4099609603       -0.4674841763       -1.8372617650
H                -5.1258603727       -0.8593665351       -1.6026781691
H                -5.7609809907        0.2541875122        0.4181105218
H                -4.9680902995        0.3213398090        3.0213964479
H                -2.6518066217       -1.1234759587        3.1217101349
H                -2.0677596305       -2.7688537809       -0.3928585618
H                -1.5752538848       -2.9240319114        1.2950594773
H                 0.5749349450       -1.0514013716       -1.4873479975
H                -1.1274110387       -1.3443109172       -1.7519520304
H                -1.8122777538        1.0598416834        0.6759279751
H                -1.8891291484        3.5937461796       -0.0339856363
H                -0.6222103591        4.0068273804       -2.3117720789
H                 1.4668844936        0.4494388380        2.9953049831
H                 5.7422058178        1.8758849195        1.1451379607
H                 6.9425900551        1.3063326755       -0.9829755487
H                 5.8822148701       -0.2520280707       -2.6073801778
H                 3.6434777230       -1.2356532776       -2.1196724784



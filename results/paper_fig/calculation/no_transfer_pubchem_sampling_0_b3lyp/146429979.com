%chk=./calculation/no_transfer_pubchem_sampling_0_b3lyp/146429979.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -3.1437318970       -2.4111757988       -0.8018858719
N                -2.9384059278       -0.9771902988       -0.9334847836
C                -3.2484350357       -0.1750680600        0.2460163971
N                -4.6505223177       -0.4256106913        0.6430630284
C                -5.0233196905       -0.0605620419        1.9930319517
C                -5.6386955105        0.0703481361       -0.2969163859
C                -2.9786544248        1.3111412315       -0.1008788340
C                -1.5756346780        1.7844343011        0.2906646438
C                -0.5161858979        0.6949085822        0.0659301414
C                -0.8272097329       -0.5185693752        0.9547915561
C                -2.2963304457       -0.5752195810        1.3840140633
C                 0.9054169027        1.2632356607        0.2586189287
N                 1.3159347740        1.5864875517       -1.0882806544
C                 2.6759110394        1.9798163466       -1.3727355825
C                 3.6776845018        0.9495940799       -0.9157834111
C                 3.6732983686       -0.3194400572       -1.4895919024
C                 4.5540741561       -1.2873935620       -1.0615204880
C                 5.4712840868       -1.0109057673       -0.0452362090
O                 6.3009802286       -2.0365102952        0.3020017664
C                 7.2632193137       -1.8128050837        1.3019560449
C                 5.4804642005        0.2523818979        0.5322027345
C                 4.5831853771        1.2155286237        0.0970507684
C                 0.5631274665        0.8845786507       -2.0073798552
O                 0.7756788135        0.8196109646       -3.1984156390
N                -0.4603271203        0.3121392993       -1.3262023994
H                -3.0558299907       -2.8601511724       -1.7896274684
H                -4.1220013748       -2.6627920982       -0.3786515298
H                -2.3679056914       -2.8347496220       -0.1666603668
H                -3.4976383078       -0.6429068368       -1.7121484377
H                -4.4679367881       -0.6543024619        2.7141160214
H                -4.8752374931        1.0074721469        2.2155565963
H                -6.0816308277       -0.2855907021        2.1219816744
H                -6.5901225401       -0.4161874027       -0.0782711466
H                -5.7895133005        1.1583329049       -0.2408054442
H                -5.3637173037       -0.1855133345       -1.3176579473
H                -3.1248534823        1.4588249855       -1.1719772385
H                -3.7072039621        1.9367751262        0.4185945231
H                -1.3151939272        2.6622434394       -0.3034133589
H                -1.5523218109        2.0728155775        1.3431591118
H                -0.2091158641       -0.4838640070        1.8542001493
H                -0.5587790256       -1.4166925620        0.3993688319
H                -2.4461586537        0.1142941702        2.2135757370
H                -2.5376395852       -1.5786421728        1.7320206079
H                 0.9135008940        2.1574207550        0.8885918113
H                 1.5773371256        0.5151164178        0.7043882776
H                 2.7281692681        2.1015694750       -2.4587351521
H                 2.8728679421        2.9437331087       -0.8911582892
H                 2.9706748933       -0.5380924208       -2.2797889226
H                 4.5592763835       -2.2720024083       -1.5017146077
H                 7.8109882090       -2.7490038472        1.3988270539
H                 7.9587060434       -1.0151287167        1.0206726489
H                 6.7976492116       -1.5665020683        2.2622683150
H                 6.1746891238        0.5017041157        1.3195251771
H                 4.6001888044        2.1935768877        0.5577258774
H                -1.2440575204       -0.1755139906       -1.7336965123



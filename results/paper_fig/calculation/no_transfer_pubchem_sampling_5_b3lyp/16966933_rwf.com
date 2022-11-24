%chk=calculation/no_transfer_pubchem_sampling_5_b3lyp/16966933_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_5_b3lyp/16966933_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 6.9831621975        1.1179306877       -2.0676528940
C                 6.2067186321        1.3675848940       -3.1213355830
C                 5.1633241431        2.4594667080       -3.2426212970
C                 4.9310154896        3.2886680326       -1.9999082140
C                 5.5865200317        4.5053453074       -1.8074084680
C                 5.3848686105        5.2788871868       -0.6607385600
C                 4.5054303620        4.8262346123        0.3179738190
C                 3.8315159589        3.6121891651        0.1560719373
C                 4.0446036815        2.8459656573       -0.9946772902
O                 3.4268092237        1.6480788229       -1.2384408480
C                 2.5180228955        1.1279977066       -0.2766749179
C                 2.0124328095       -0.2074169934       -0.8146811107
C                 1.0733275101       -0.9536502461        0.1454583624
C                -0.2947207279       -0.2745781583        0.3307000064
N                -1.1888475172       -1.0217637265        1.2054559267
C                -2.1519072254       -1.9517644578        0.8453988106
C                -2.4552092836       -2.2654641211       -0.6014424882
N                -3.3999088461       -3.3600188081       -0.7612825270
C                -2.8619707238       -4.7070616262       -0.5663925547
C                -4.7624213611       -3.2289260610       -0.6284365699
O                -5.4854038933       -4.2088125394       -0.4583516339
C                -5.4363497470       -1.8721411343       -0.7409119705
C                -6.6857327777       -1.7463513729       -0.1185261634
C                -7.3821573477       -0.5493365554       -0.2482519865
C                -6.8231810543        0.4754006163       -1.0124039127
C                -5.5813627458        0.2524710694       -1.6052570828
N                -4.8929316851       -0.8876063881       -1.4734793096
N                -2.7732510314       -2.4827043969        1.8697188387
C                -2.2008718681       -1.8912901746        2.9873235097
C                -2.4801014424       -2.0807595249        4.3454493993
C                -1.7491452671       -1.3444271712        5.2735709571
C                -0.7560320589       -0.4324183505        4.8663865737
C                -0.4620001948       -0.2298869698        3.5189372155
C                -1.1998654500       -0.9724763425        2.5947779912
H                 6.9232272856        1.7062388827       -1.1560104019
H                 7.7106693858        0.3108702214       -2.0828867704
H                 6.3103462286        0.7473996494       -4.0132086225
H                 5.4582201116        3.1258551562       -4.0665135052
H                 4.2191366015        1.9968168935       -3.5579703380
H                 6.2709740648        4.8526068424       -2.5782775494
H                 5.9084320908        6.2226303365       -0.5391026487
H                 4.3330405565        5.4132206283        1.2162439362
H                 3.1489003624        3.2745707955        0.9275089003
H                 1.6951476253        1.8404621149       -0.1187376518
H                 3.0278887772        0.9846347381        0.6880661074
H                 1.5133160554       -0.0347936221       -1.7773790205
H                 2.8862767156       -0.8342544212       -1.0257301569
H                 1.5497195071       -1.0739206953        1.1271577140
H                 0.9060172561       -1.9674475325       -0.2383162829
H                -0.7875524629       -0.1485447292       -0.6389564594
H                -0.1826086614        0.7284668228        0.7564706370
H                -1.5314472821       -2.5451368839       -1.1249687299
H                -2.8553265704       -1.3828725667       -1.1064445851
H                -3.5622299730       -5.4315516220       -0.9797145945
H                -1.9021583985       -4.7811363878       -1.0875732225
H                -2.7165550908       -4.9227552016        0.4985011433
H                -7.0818638138       -2.5876037434        0.4368716787
H                -8.3479297931       -0.4196611912        0.2325375520
H                -7.3333917814        1.4242670091       -1.1482314479
H                -5.1139740444        1.0229841123       -2.2161700415
H                -3.2480917048       -2.7843781390        4.6518863478
H                -1.9456702395       -1.4730423669        6.3343440667
H                -0.2046550549        0.1250229819        5.6185847547
H                 0.3077639499        0.4736855705        3.2138292252



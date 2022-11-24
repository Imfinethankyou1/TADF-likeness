%chk=calculation/no_transfer_pubchem_sampling_5_b3lyp/4042596_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_5_b3lyp/4042596_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 6.7036730310       -1.6588472253        2.8898728299
C                 6.3480265746       -1.6011511686        1.3993395729
C                 6.7797085394       -0.3066421239        0.6885442245
C                 6.1257403522        0.9898396460        1.1979011278
C                 4.6133563166        1.0562320242        1.0223024496
O                 4.3272132426        0.9685860886       -0.3779937019
C                 3.0372601832        0.9945749784       -0.7962964029
C                 2.8448342742        0.8843380262       -2.1853416570
C                 1.5658077161        0.9003336088       -2.7121692766
C                 0.4451370561        1.0304260503       -1.8716301971
C                -0.8945144878        1.0371748983       -2.4992299901
O                -1.1032280705        0.8924783303       -3.6881419948
O                -1.8960316613        1.2379016743       -1.5887585821
C                -3.2465816886        1.2160573160       -2.1005894778
C                -3.8850701505       -0.1549802209       -1.9731134819
C                -3.9006338897       -0.9072601821       -3.1081583093
C                -4.5640957344       -2.1912956277       -3.2080590447
O                -4.5739871702       -2.9396466341       -4.1562248368
O                -5.3020124576       -2.5482555648       -2.0797690587
C                -5.2945359884       -1.8226366375       -0.9296848442
C                -6.1387363080       -2.3491437873        0.0750853962
C                -6.2350208350       -1.7056478453        1.2758766604
C                -5.4512140368       -0.5503109163        1.5511106342
C                -5.5243607400        0.0612301885        2.8280508305
C                -4.7292648109        1.1371753893        3.1511359311
C                -3.8063789820        1.6136536423        2.1994578493
C                -3.7197145381        1.0462493702        0.9426751575
C                -4.5625371150       -0.0293438049        0.5492782013
C                -4.5459278635       -0.6490471178       -0.7655610710
C                 0.6434303182        1.1390292552       -0.4900166850
C                 1.9267005882        1.1215920621        0.0520712187
H                 7.7803435161       -1.5143432948        3.0444460197
H                 6.4335854223       -2.6292857855        3.3212976410
H                 6.1815850789       -0.8884918908        3.4696651490
H                 6.8279991977       -2.4449778960        0.8865251635
H                 5.2679026259       -1.7567853461        1.2722212470
H                 6.5712365665       -0.4017012045       -0.3835676920
H                 7.8689753077       -0.1974658504        0.7833967544
H                 6.5682156497        1.8452466046        0.6723575125
H                 6.3392731194        1.1373938145        2.2645358815
H                 4.2227110371        2.0037177740        1.4186357795
H                 4.1107818406        0.2350942965        1.5510422627
H                 3.7180300500        0.7853038317       -2.8221371013
H                 1.4063851599        0.8121088307       -3.7816549040
H                -3.2206925088        1.5061239812       -3.1505469758
H                -3.7942174671        1.9686558360       -1.5319976705
H                -3.4125404172       -0.5600945555       -4.0113994205
H                -6.7012673799       -3.2481428445       -0.1521245480
H                -6.8976515803       -2.0844767312        2.0495760869
H                -6.2185120365       -0.3512506693        3.5563252812
H                -4.7922056830        1.5965914442        4.1333602853
H                -3.1384849150        2.4310734566        2.4581701414
H                -2.9496308096        1.4067968236        0.2780626122
H                -0.2112647503        1.2311168263        0.1706606357
H                 2.0523993121        1.2051308564        1.1249733863



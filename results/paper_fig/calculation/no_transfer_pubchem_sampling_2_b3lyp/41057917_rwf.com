%chk=calculation/no_transfer_pubchem_sampling_2_b3lyp/41057917_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_2_b3lyp/41057917_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 7.9482369732        2.6270903533       -1.2084852978
O                 6.7202181887        1.9628092045       -1.4549470473
C                 5.9742238062        1.5620073088       -0.3841878316
C                 6.3447389424        1.7521632914        0.9522664523
C                 5.4879897906        1.2911897440        1.9532453657
C                 4.2899870853        0.6544137745        1.6495415539
C                 3.9274438414        0.4599657584        0.3020875551
N                 2.7124663278       -0.2055571495        0.0022978664
C                 2.0192450217       -1.0251310932        1.0338921309
C                 1.1362763382       -1.8595876043        0.1258231901
C                 1.8222279675       -3.0123601533       -0.6015351823
C                 0.8421896570       -3.6861560518       -1.2056899953
C                -0.4155856650       -2.9087737156       -0.8797501108
O                -0.1364679171       -2.4302869718        0.4733252724
C                -0.3308722704       -1.5762373797       -1.7698768252
C                -1.6867456915       -0.8816626278       -1.9369692197
O                -2.1027023317       -0.5571306325       -3.0421821767
N                -2.3643799791       -0.6919033840       -0.7579257740
C                -3.6278332009       -0.0933209169       -0.5638001002
C                -4.4238828604        0.4399511473       -1.5839464861
C                -5.6592977095        1.0120614865       -1.2936524993
C                -6.1246107882        1.0628788233        0.0220021495
O                -7.3502103921        1.6431623145        0.2068529624
C                -7.8677571715        1.7230837240        1.5219539986
C                -5.3434139040        0.5357874723        1.0601793691
C                -4.1047391443       -0.0354712534        0.7619295520
O                -3.2622088027       -0.5756550855        1.7021784470
C                -3.6757198905       -0.6016500318        3.0576501250
C                 0.6744184661       -0.7944625726       -0.8978946513
C                 1.9970327470       -0.1209809837       -1.2152210675
O                 2.3735910895        0.4113160529       -2.2386179891
C                 4.7758795689        0.9156313120       -0.7121339409
H                 8.6507753105        1.9887830022       -0.6558556986
H                 7.7983408224        3.5621510930       -0.6524126731
H                 8.3662579667        2.8557652505       -2.1905527243
H                 7.2710985306        2.2468940549        1.2182717262
H                 5.7605587618        1.4393072599        2.9948016899
H                 3.6381917031        0.3321326581        2.4533588876
H                 1.4278642323       -0.4072203397        1.7233132476
H                 2.7509147862       -1.6014154066        1.6065128196
H                 2.8937399631       -3.1631079467       -0.6532710442
H                 0.9055712628       -4.5241443260       -1.8894944992
H                -1.3719332780       -3.4291340369       -0.9010274558
H                 0.0295472584       -1.7941162927       -2.7754765415
H                -1.9299797310       -1.0450981160        0.0896174234
H                -4.0602263791        0.3973273427       -2.6009738435
H                -6.2773519641        1.4256182245       -2.0834567886
H                -8.0078196492        0.7274331976        1.9654999641
H                -8.8385410410        2.2147606279        1.4338127434
H                -7.2201979400        2.3207020137        2.1785621191
H                -5.6893095694        0.5696414745        2.0839733566
H                -2.8692096815       -1.0899284452        3.6075559424
H                -4.6018380222       -1.1774844741        3.1833164325
H                -3.8231551227        0.4122662983        3.4522089831
H                 0.1109356703       -0.0485783458       -0.3200007170
H                 4.5160290177        0.7902600715       -1.7528981453



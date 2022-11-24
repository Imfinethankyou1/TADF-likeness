%chk=calculation/no_transfer_pubchem_sampling_6_b3lyp/3930306_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_6_b3lyp/3930306_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -0.1300554352        5.1941921427        0.9167002036
O                -1.1465215869        4.2013241516        0.8627688302
C                -0.8981030913        3.0754624165        0.1342352734
C                -1.8495870093        2.0492744893        0.2424834234
C                -1.6617234547        0.8493929748       -0.4312513736
C                -0.5324703229        0.6428791475       -1.2386686114
C                -0.2465126772       -0.7165214912       -1.8907543836
C                -1.4726308649       -1.3687237039       -2.5038885258
C                -2.0775036380       -0.8042978666       -3.6586854575
C                -3.2004119409       -1.3543955381       -4.2236643354
C                -3.7999654252       -2.5155995492       -3.6619085356
C                -4.9635291280       -3.1115463841       -4.2145617230
C                -5.5180303315       -4.2383446225       -3.6489782024
C                -4.9302992272       -4.8220406014       -2.5020828034
C                -3.7999511324       -4.2718181385       -1.9373977563
C                -3.2124037454       -3.1098214907       -2.5009112003
C                -2.0438140804       -2.4983526213       -1.9569688886
O                -1.5284166954       -3.1347809629       -0.8383584100
C                -0.3121162714       -2.7568650819       -0.3710913703
N                 0.1325949806       -3.5664697781        0.6263982937
C                 1.2937178675       -3.3098892466        1.1541207473
N                 2.0616225481       -2.2785789888        0.7101556857
N                 3.2863019459       -1.8844773048        1.1676374259
C                 3.5392187470       -0.8411499297        0.3762440810
C                 4.8168979857       -0.0744565763        0.5186108538
S                 4.6075784881        1.7569280875        0.6010928835
C                 3.7738326548        1.9511716707        2.1313865827
N                 3.0947351816        2.9954616704        2.4865308156
C                 2.7119610438        2.7151033381        3.8022139279
C                 1.9638573629        3.4519409512        4.7230418482
C                 1.7345611142        2.8734800452        5.9720160187
C                 2.2306473338        1.5988512816        6.2984026789
C                 2.9785970951        0.8492233094        5.3852895160
C                 3.1963115018        1.4492883238        4.1562888339
O                 3.8945764401        0.9566562376        3.0732159616
N                 2.5896120935       -0.5320329220       -0.5459610404
C                 1.6433814612       -1.4386293657       -0.3185595540
C                 0.3660013278       -1.6646549145       -0.8788500268
C                 0.3888951433        1.6844384773       -1.3637148664
C                 0.2197290380        2.8935018034       -0.6870240087
H                 0.0346771553        5.6570821795       -0.0655957948
H                 0.8163572136        4.7776484048        1.2849820896
H                -0.4936657063        5.9542384618        1.6112135796
H                -2.7175047184        2.2101850061        0.8745067217
H                -2.4022099023        0.0606950580       -0.3236190413
H                 0.4925411221       -0.5411443665       -2.6842412298
H                -1.6329252401        0.0917729272       -4.0837359269
H                -3.6476635729       -0.9060961153       -5.1069115966
H                -5.4106960488       -2.6595971545       -5.0966508347
H                -6.4094368762       -4.6831374839       -4.0828607784
H                -5.3744081537       -5.7117234975       -2.0641486383
H                -3.3462852000       -4.7168348449       -1.0589933542
H                 1.6962070785       -3.9061198980        1.9663519258
H                 5.4408729437       -0.2045914816       -0.3719590828
H                 5.3714338614       -0.4295142867        1.3877464268
H                 1.5785883652        4.4340359819        4.4689527395
H                 1.1568008273        3.4192285297        6.7124827019
H                 2.0293291265        1.1854289083        7.2823311740
H                 3.3665572389       -0.1355083810        5.6223101725
H                 1.2807080302        1.5407898956       -1.9659495034
H                 0.9701381596        3.6680367172       -0.7912285617



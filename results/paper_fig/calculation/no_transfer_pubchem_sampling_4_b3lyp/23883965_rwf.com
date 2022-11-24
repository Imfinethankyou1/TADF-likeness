%chk=calculation/no_transfer_pubchem_sampling_4_b3lyp/23883965_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_4_b3lyp/23883965_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -0.5244038313        0.0163236025       -2.1468870912
C                 0.2155438297        0.2662572802       -0.8616176197
N                -0.3322857141        0.5096279567        0.2854230567
N                -1.7314931323        0.3853256576        0.3651719197
C                -2.4929492589        1.3714379952        1.0280359462
C                -2.0278903896        2.7598055778        1.2246748645
C                -2.9405773969        3.8103302383        1.0055827992
C                -2.5842145096        5.1328581116        1.2396103981
C                -1.2943609809        5.4082433695        1.6886481352
N                -0.9051219815        6.8046586110        1.9311742059
O                 0.2421213132        7.0188623497        2.3228621018
O                -1.7504165906        7.6768784891        1.7289418107
C                -0.3639186536        4.3949687088        1.9096380792
C                -0.7331361586        3.0741989140        1.6787570746
C                -3.7156530329        0.9256107761        1.3953588774
S                -3.9810277036       -0.7647718671        1.0189867372
C                -2.3210479595       -0.8859563728        0.3216516165
N                -1.7441428717       -1.9224881630       -0.1389131346
C                -2.3740422526       -3.1892301518       -0.0842917492
C                -3.2311722808       -3.6060981462       -1.1257149151
C                -3.5357252323       -2.6958477394       -2.2919236321
C                -3.7785042526       -4.8923272272       -1.0654585671
C                -3.4829082355       -5.7547086040       -0.0119440420
C                -2.6206642853       -5.3352494661        0.9995556809
C                -2.0521977768       -4.0583142024        0.9810067300
C                -1.1091085055       -3.6097764942        2.0708429400
C                 1.7020139939        0.2769681902       -0.8685676558
C                 2.4202077865        0.2884633605        0.3423750334
C                 3.8071792249        0.3171360468        0.3579210687
C                 4.5246204732        0.3076743696       -0.8466179608
N                 5.9388387371        0.3296913166       -0.8375765062
C                 6.7829754000       -0.2593787162        0.0957380098
C                 8.0536741779        0.0485849929       -0.3094721946
N                 8.0313272525        0.8205747818       -1.4532072318
C                 6.7616926067        0.9666109064       -1.7434202173
C                 3.8286272384        0.2691916893       -2.0593889909
C                 2.4369607039        0.2613300759       -2.0652386662
H                -1.5850219706        0.2402250490       -2.0333032331
H                -0.4252576360       -1.0316632550       -2.4457537835
H                -0.1206723333        0.6447846550       -2.9468966908
H                -3.9320318347        3.5846585957        0.6266277896
H                -3.2773515786        5.9470120516        1.0680044321
H                 0.6272137251        4.6494846344        2.2643382485
H                -0.0211651756        2.2779339456        1.8503586001
H                -4.4812001705        1.5059352694        1.8887277430
H                -4.0655046941       -1.7873329824       -1.9796937290
H                -4.1629217847       -3.2074583454       -3.0285033419
H                -2.6170640804       -2.3723056793       -2.7956435976
H                -4.4416700971       -5.2184224163       -1.8634969954
H                -3.9145823571       -6.7515073442        0.0157420393
H                -2.3776949945       -6.0073284601        1.8194653118
H                -1.5530686596       -2.8230987652        2.6946293993
H                -0.8463766754       -4.4450593580        2.7275108763
H                -0.1874266144       -3.1929572700        1.6492411366
H                 1.8698685477        0.2737814910        1.2756652185
H                 4.3397793762        0.3583381082        1.3025653799
H                 6.4005548035       -0.8619965060        0.9048326181
H                 8.9868920063       -0.2442827743        0.1522355363
H                 6.3607869453        1.5482986503       -2.5620243285
H                 4.3745742345        0.2193925050       -2.9957109099
H                 1.9305212674        0.2261019843       -3.0230296307



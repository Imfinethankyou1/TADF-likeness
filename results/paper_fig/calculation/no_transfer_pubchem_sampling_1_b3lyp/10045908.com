%chk=./calculation/no_transfer_pubchem_sampling_1_b3lyp/10045908.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -7.1564484082        1.9174920054       -1.7358587703
C                -6.8212676203        1.6123450243       -0.2820705395
O                -5.4139960772        1.5772102831       -0.0724106710
C                -4.7842622128        0.4454212752       -0.3578381838
O                -5.3396737241       -0.5615026738       -0.7494927287
C                -3.3233016739        0.5163370553       -0.1320624250
C                -2.6418617401       -0.6503759184       -0.3108240552
N                -1.3166664843       -0.8319401840       -0.1646566401
C                -0.6954078122       -2.1630341324       -0.3569363591
C                -0.0619247641       -2.5941189072        0.9694270582
C                 0.3434607087       -2.0650619832       -1.4783254939
C                -1.7157891707       -3.2310627231       -0.7583103980
C                -0.5585126001        0.2706774794        0.1937990224
N                 0.7596736385        0.1172284852        0.3264104237
C                 1.5494289901        1.1369493471        0.6651031901
N                 2.8656376874        0.8745194472        0.7498869093
C                 3.3950109178       -0.4563776691        0.5404600394
C                 4.8249334780       -0.3805252960        1.0747718992
C                 5.7519475857       -1.4279879829        0.4766139068
C                 5.2200692088        1.0882742130        0.7728009101
N                 5.6691888523        1.3649685834       -0.5741818687
C                 3.9098575173        1.8396518687        1.0358193723
C                 0.9754610274        2.4097550154        0.9145099989
F                 1.7446877519        3.4588656955        1.3002449560
C                -0.3688330068        2.5856139961        0.7774873218
C                -1.1852307705        1.5088488811        0.3994939442
C                -2.6292384787        1.7374159416        0.2504501614
O                -3.1341934026        2.8320886157        0.4320173905
H                -8.2322003870        2.0165419704       -1.8515311371
H                -6.8047301434        1.1101565609       -2.3737517809
H                -6.6760171189        2.8441206614       -2.0399129884
H                -7.1852358608        2.4020791436        0.3781625910
H                -7.2514020028        0.6478651662        0.0107974810
H                -3.2375967963       -1.5018047411       -0.5968934968
H                 0.6641013362       -1.8588245008        1.2966649947
H                -0.8303095082       -2.6923516420        1.7332143648
H                 0.4330955872       -3.5535717530        0.8484865395
H                 1.1056918601       -1.3390576606       -1.2214131560
H                 0.8102309823       -3.0338476428       -1.6326828004
H                -0.1391081681       -1.7597165241       -2.4041319527
H                -1.1836193732       -4.1713651408       -0.8826054583
H                -2.4722065543       -3.3803842082        0.0081390565
H                -2.1971007535       -2.9967395763       -1.7045542032
H                 3.3866824163       -0.7086064092       -0.5281341160
H                 2.7846601134       -1.1942363074        1.0624989389
H                 4.8043215759       -0.5002326458        2.1620381834
H                 6.7631703822       -1.2987958899        0.8564408737
H                 5.7751943795       -1.3587394275       -0.6093664341
H                 5.4102552595       -2.4251770739        0.7431614409
H                 6.0102739460        1.4287519542        1.4474095936
H                 6.5523418659        0.9093419487       -0.7673829630
H                 4.9868286671        1.0514229118       -1.2550603263
H                 3.8414780061        2.7107409235        0.3814435701
H                 3.8492682113        2.1764325940        2.0758163842
H                -0.8308143401        3.5443225660        0.9568154301



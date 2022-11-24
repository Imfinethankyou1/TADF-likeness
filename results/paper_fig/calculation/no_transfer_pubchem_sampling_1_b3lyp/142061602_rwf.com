%chk=calculation/no_transfer_pubchem_sampling_1_b3lyp/142061602_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_1_b3lyp/142061602_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -0.3528755820        2.2617740808       -0.8970534944
C                 0.0054911718        0.8493962681       -1.3783027111
C                -0.5636553252        0.5289709156       -2.7673782133
N                -0.3135800406       -0.2084697626       -0.3817845137
C                 0.7915727576       -1.0637956165        0.0577711558
C                 1.7151121714       -0.4009860888        1.0686579369
C                 1.1679935048        0.1985647109        2.2124323802
C                 1.9762720499        0.8192606992        3.1615036891
C                 3.3610601178        0.8570633687        2.9832278003
C                 3.9224410966        0.2573510805        1.8594474464
C                 3.1096301732       -0.3731281472        0.9106491269
N                 3.7476050152       -1.0431810504       -0.1978615829
C                 4.2678184512       -2.3921192567        0.0211479379
C                 3.9586339918       -0.3336876119       -1.3647640447
O                 3.4779189082        0.7547408607       -1.5980071041
O                 4.7413346124       -0.9474957027       -2.2993689638
C                -1.5522045002       -0.4860765858        0.1343486961
O                -1.7511583125       -1.3216203051        1.0119447307
O                -2.5204717438        0.2700879984       -0.4405584578
C                -3.9236391826        0.1525700669       -0.0119852260
C                -4.4476005152       -1.2585031322       -0.3015772089
C                -4.0626375456        0.5332361589        1.4662960021
C                -4.6312049501        1.1806169762       -0.9000280127
H                -1.4331263944        2.4126205526       -0.8412601779
H                 0.0815234845        2.4518142342        0.0898958427
H                 0.0641631891        2.9961213112       -1.5954079960
H                 1.0958360578        0.8179785770       -1.4534189447
H                -1.6558672431        0.5677806352       -2.7695934672
H                -0.1878461139        1.2543872270       -3.4978633173
H                -0.2511041577       -0.4692926969       -3.0948305704
H                 0.3447747007       -1.9566833831        0.5014253459
H                 1.3622412556       -1.3711817093       -0.8227554330
H                 0.0920902615        0.1589765511        2.3537627224
H                 1.5262670478        1.2734347828        4.0401877777
H                 3.9984363751        1.3471906299        3.7138039996
H                 4.9970107148        0.2786548884        1.6996957440
H                 5.3560315371       -2.4092192067        0.1883746742
H                 3.7926271274       -2.8052042019        0.9120026782
H                 4.0196467941       -3.0530792560       -0.8190302270
H                 5.1829213269       -1.7338666682       -1.9437243194
H                -5.5224005486       -1.3017607795       -0.0910358871
H                -4.2984236440       -1.5090257059       -1.3578143181
H                -3.9358940090       -1.9994572459        0.3136735603
H                -3.6392000502        1.5282390393        1.6437042864
H                -5.1237702147        0.5630921772        1.7390360472
H                -3.5539202721       -0.1883790776        2.1065978358
H                -5.7045383612        1.1881212327       -0.6824464132
H                -4.4952413781        0.9379029054       -1.9589938007
H                -4.2360938098        2.1862672619       -0.7227450114



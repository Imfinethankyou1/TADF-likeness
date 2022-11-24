%chk=calculation/no_transfer_pubchem_sampling_4_b3lyp/20547649_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_4_b3lyp/20547649_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 6.1916517331        1.8192922166        1.5891728714
C                 5.1581481269        1.6381013087        0.4715035674
C                 4.4060995390        0.3032509127        0.5834960591
C                 3.3110069250        0.0329359380       -0.4753649141
C                 3.8295641198       -0.0519935474       -1.9343909087
C                 4.7963438669       -1.2049256004       -2.2938904119
C                 6.2890545137       -0.8908465533       -2.1233563816
C                 2.5651609541       -1.2816648339       -0.1082931091
O                 1.4886545802       -1.0167307446        0.7941294035
C                 0.4140924789       -0.2988636998        0.1829395655
C                -0.2724810096        0.4431153346        1.3491035380
C                -1.1039274662        1.6657108447        0.9305061933
N                -1.8059414352        2.2589357348        2.0643826587
C                -3.0079699966        1.8373593501        2.5997670674
C                -3.2352618575        2.6498884466        3.6834133119
N                -2.2106739279        3.5573604740        3.8361421452
C                -1.3757832293        3.2895141298        2.8536395256
C                -0.4865819886       -1.2986064475       -0.6250306806
C                -2.0025677435       -1.1576174074       -0.5747760005
C                -2.6774577374       -0.3957333916       -1.5550064580
C                -1.9041787225        0.3635066102       -2.6204392110
C                -4.0844177008       -0.3279157212       -1.5493060359
C                -4.8079608232        0.4243318591       -2.6516510655
C                -4.8200114778       -0.9700451657       -0.5333525664
C                -6.3268848848       -0.7965334718       -0.4632843347
C                -4.1522682288       -1.7638646825        0.4184924454
C                -4.9399059122       -2.5542564563        1.4479867087
C                -2.7437858652       -1.8486679920        0.4083824471
C                -2.0676531280       -2.7026804393        1.4687136578
O                 0.9466216630        0.6269514435       -0.7718431147
C                 2.2275253553        1.1431741116       -0.3963626663
H                 6.9445466833        1.0217099663        1.5654800953
H                 5.7161635032        1.7942904897        2.5772159789
H                 6.7173764651        2.7761437912        1.4955357561
H                 5.6632655998        1.7046333001       -0.5004427487
H                 4.4465116258        2.4745131721        0.5031692033
H                 3.9259464574        0.2496940213        1.5718035977
H                 5.1339120220       -0.5192394664        0.5634551233
H                 2.9358217496       -0.1348382341       -2.5674938157
H                 4.2981153877        0.9030141800       -2.2096556987
H                 4.6286355603       -1.4720835822       -3.3453004519
H                 4.5443894406       -2.1076188004       -1.7211994752
H                 6.5582381835       -0.6794291860       -1.0835903968
H                 6.5742707174       -0.0173862219       -2.7226607830
H                 6.9042488899       -1.7347407484       -2.4566922034
H                 3.2197017178       -1.9810768741        0.4189607037
H                 2.1918370679       -1.7844962819       -1.0103875445
H                 0.5075499354        0.7644428229        2.0470745719
H                -0.9074537748       -0.2568280399        1.8981463043
H                -0.4605679268        2.4341154252        0.4949685087
H                -1.8467325444        1.3982309970        0.1772604729
H                -3.5685891175        1.0282025660        2.1540845965
H                -4.0820955089        2.6401991198        4.3570519824
H                -0.4478326789        3.8101158914        2.6519620740
H                -0.1470748405       -1.2557191425       -1.6616238077
H                -0.2140017104       -2.2917990611       -0.2644221467
H                -2.3837928174        1.3187978745       -2.8535943420
H                -0.8830323613        0.5892960967       -2.3121084238
H                -1.8491104060       -0.2013866806       -3.5627920808
H                -4.2593845491        0.3770027776       -3.5955838092
H                -4.9437837356        1.4891261314       -2.4118411431
H                -5.8001955053        0.0081188523       -2.8410136371
H                -6.6917693508       -0.8512932731        0.5651646710
H                -6.6377760619        0.1747459623       -0.8554771625
H                -6.8638503556       -1.5646442938       -1.0386749785
H                -5.9555779338       -2.7683972415        1.1094657672
H                -5.0218200244       -2.0236304232        2.4078638456
H                -4.4670378970       -3.5171011940        1.6598084715
H                -0.9955723854       -2.5229886532        1.5493812368
H                -2.2005389700       -3.7749702424        1.2659655210
H                -2.4953341513       -2.5166232466        2.4597877287
H                 2.1833095445        1.5709967505        0.6140417617
H                 2.4308753356        1.9564211391       -1.0985126300



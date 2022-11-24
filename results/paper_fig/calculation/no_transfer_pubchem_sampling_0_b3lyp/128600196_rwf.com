%chk=calculation/no_transfer_pubchem_sampling_0_b3lyp/128600196_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_0_b3lyp/128600196_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 5.3809398905        1.3412079994        1.8859411559
C                 5.1479961158        0.5935780174        0.5850601626
C                 4.9525092889       -0.9347661337        0.7110174057
C                 3.9234484527       -1.2724043478       -0.3959800569
C                 3.6032679182        0.1093507425       -1.0326210134
C                 2.2056540352        0.3099821288       -1.5161187789
N                 1.8927715569        0.4305660515       -2.7851287166
N                 0.5194626684        0.5221525847       -2.8759237885
C                 0.0423209364        0.4665954954       -1.6500370145
N                -1.2906864311        0.5948809444       -1.2667117986
C                -2.1338856248        1.2930476802       -2.2640834909
C                -3.6051406131        0.9339292592       -2.0293910677
O                -3.8940990891        0.8610309641       -0.6437302113
C                -3.2357711883       -0.2396992608        0.0211371021
C                -4.2045493263       -1.4429161965        0.1103463842
C                -5.0527785387       -1.1522629976        1.3544081003
C                -4.0067611183       -0.6343158604        2.3574777087
C                -3.0424849505        0.2201123507        1.4982539109
C                -1.9356263405       -0.6108474767       -0.7251747092
N                 1.0678069722        0.3334306207       -0.7372145347
C                 0.9589618559        0.4106021739        0.7120145774
O                 3.9379939233        1.0671899834       -0.0186219305
H                 6.3179967928        1.0151274507        2.3508187331
H                 4.5625050790        1.1529553553        2.5902094773
H                 5.4425565105        2.4192946864        1.7073253516
H                 5.9811014822        0.7944962733       -0.1102328741
H                 5.8976617030       -1.4733988388        0.5918119271
H                 4.5555228238       -1.1825292184        1.7018328342
H                 4.3098948650       -1.9644266973       -1.1493636212
H                 3.0213827551       -1.7246601748        0.0273990131
H                 4.2452648899        0.2656946286       -1.9110145470
H                -1.8402658357        1.0253409394       -3.2867135777
H                -1.9781767668        2.3700268422       -2.1385067474
H                -4.2517007720        1.7123272553       -2.4442934575
H                -3.8642560426       -0.0095942518       -2.5347375677
H                -4.7822528075       -1.5601463682       -0.8128366536
H                -3.6294205748       -2.3688200830        0.2606868382
H                -5.7790462380       -0.3633672692        1.1245198497
H                -5.6067714883       -2.0246113258        1.7170551873
H                -4.4454711999       -0.0677611409        3.1854075347
H                -3.4715074523       -1.4857471902        2.7985687034
H                -2.0029296329        0.1211829240        1.8268607104
H                -3.3001727114        1.2813871609        1.5480715570
H                -1.2523307122       -1.1078406255       -0.0314326024
H                -2.1482992186       -1.3378419577       -1.5277609523
H                 0.8259459717       -0.5817893927        1.1596521893
H                 1.8686538002        0.8645027576        1.1041080159
H                 0.0927643861        1.0297515373        0.9576472824



%chk=./calculation/no_transfer_pubchem_sampling_0_b3lyp/82981446.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -2.0003034557       -2.6516014132        0.8416284622
O                -2.2952181065       -2.0309778875       -0.3838682536
C                -3.0527664085       -0.8530286598       -0.2727179624
C                -2.3488292846        0.3170729169        0.4480679500
C                -3.1715651943        1.5977793024        0.2244749018
N                -0.9520509938        0.4031408908        0.0666116297
C                -0.7035586351        0.5464064660       -1.3508881924
C                -0.1149539030        1.1901326971        0.9610438378
C                 0.3822979968        2.5223889583        0.3926146256
C                 1.0724417240        0.3294434302        1.4723197003
C                 2.0057207387       -0.1310532492        0.3939415585
C                 3.1472274596        0.5947749224        0.0728161416
C                 3.9977619457        0.1820835816       -0.9344261435
C                 3.7192525269       -0.9806967027       -1.6432604222
O                 4.5795955802       -1.3605622682       -2.6317230954
C                 2.5854427903       -1.7184982481       -1.3260484115
C                 1.7418211845       -1.2960515327       -0.3161806293
H                -1.5342944053       -3.6051265386        0.5976589424
H                -2.9133555524       -2.8335716395        1.4233012315
H                -1.3078883892       -2.0537818385        1.4422383481
H                -3.2926310566       -0.5724216361       -1.3007335902
H                -3.9973330317       -1.0592320613        0.2568332683
H                -2.3591801097        0.0976656241        1.5239790568
H                -3.2106944861        1.8515683972       -0.8316246746
H                -2.7321582096        2.4297308425        0.7695767611
H                -4.1886971708        1.4522797721        0.5796625145
H                 0.3685810830        0.5569332045       -1.5237404428
H                -1.1124282187       -0.3212883646       -1.8649870980
H                -1.1363452644        1.4570281881       -1.7894462702
H                -0.7155688474        1.4221454901        1.8500371214
H                -0.4476747778        3.1048375111       -0.0003457318
H                 0.8581338130        3.0974875802        1.1839331267
H                 1.1067460693        2.3686656129       -0.4017227403
H                 1.6217845071        0.9198277288        2.2085448176
H                 0.6524889839       -0.5427294419        1.9770042019
H                 3.3801144728        1.4960531016        0.6217409493
H                 4.8833185768        0.7446013033       -1.1848481966
H                 4.2655584691       -2.1787798353       -3.0345186402
H                 2.3645123201       -2.6260116821       -1.8733691228
H                 0.8546972592       -1.8666335230       -0.0902085295



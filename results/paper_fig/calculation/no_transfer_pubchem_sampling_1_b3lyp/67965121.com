%chk=./calculation/no_transfer_pubchem_sampling_1_b3lyp/67965121.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -1.4032841708       -1.8682174902       -2.5215766409
O                -2.2583614474       -1.0559206245       -1.7398098802
C                -2.1656983418       -1.3071278039       -0.3812126683
C                -3.1020366740       -2.1502304841        0.2023984126
C                -3.0645566144       -2.4022760077        1.5607970494
C                -2.0819694505       -1.8137691854        2.3366119581
C                -1.1541735626       -0.9677306816        1.7557528609
C                -1.1765680030       -0.6941674484        0.3940128935
C                -0.1931194182        0.2431189626       -0.2729782781
C                 0.1020493125        1.4955343476        0.5763213896
C                -1.1521165859        2.2737565577        0.9893109602
C                -0.7512113005        3.4450192073        1.8864084620
C                -1.9298213800        2.7855316366       -0.2210375721
N                 0.9868641185       -0.5069380988       -0.7004441554
C                 1.7240008296       -1.1887243188        0.3495201156
C                 3.0081575224       -1.7766731679       -0.2232850587
C                 3.9752007960       -0.6583450797       -0.6316445838
C                 3.1946484069        0.5915857105       -1.0539978189
C                 1.8486113678        0.1836267471       -1.6524588411
H                -1.5096172698       -1.5204410847       -3.5477780482
H                -0.3633511448       -1.7687428334       -2.1966201048
H                -1.7028737578       -2.9211012448       -2.4602542242
H                -3.8599937338       -2.5960559302       -0.4244726545
H                -3.7965095403       -3.0569106176        2.0101437524
H                -2.0398227050       -2.0084835845        3.3982357439
H                -0.4020942251       -0.5108474845        2.3805398626
H                -0.6662166168        0.5785270429       -1.2030979662
H                 0.6508114746        1.2146292962        1.4771241260
H                 0.7466021440        2.1581671513       -0.0071517791
H                -1.8065173082        1.6071092270        1.5611779213
H                -0.1058629275        4.1346588017        1.3453547030
H                -0.2161794600        3.0936021903        2.7663454358
H                -1.6315114916        3.9904368997        2.2188183602
H                -1.2841555093        3.3760770951       -0.8687289813
H                -2.7544398738        3.4159924857        0.1046691170
H                -2.3450146234        1.9622230925       -0.7960556695
H                 1.0876327939       -1.9904798648        0.7384890193
H                 1.9854990548       -0.5285373037        1.1928473544
H                 3.4734977253       -2.4277484694        0.5180225465
H                 2.7499225657       -2.3890030937       -1.0882797199
H                 4.6021457290       -1.0044192513       -1.4550456937
H                 4.6352146424       -0.4106289088        0.2018583049
H                 3.0208849931        1.2343615204       -0.1893314826
H                 3.7630167375        1.1683004445       -1.7853237916
H                 1.3281916520        1.0606163810       -2.0428951388
H                 2.0341242702       -0.4893617353       -2.5012785974



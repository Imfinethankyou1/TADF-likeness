%chk=./calculation/no_transfer_pubchem_sampling_3_b3lyp/54144807.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 4.9507643409       -1.9282090373        0.8846103723
C                 6.4062565143       -1.4927391088        0.7674497724
C                 6.7130568962       -0.6854050714       -0.4929344697
C                 6.0840740002        0.7141322391       -0.5185291251
N                 6.6878319366        1.4759019973       -1.6104301451
C                 4.6185059856        0.6800547895       -0.7872280544
N                 3.7337495660        1.2975431598       -0.0987746980
C                 2.4776278325        1.1377722967       -0.5889247135
C                 1.3419617658        1.8326140801        0.0782057989
O                 1.4570637918        2.9088553894        0.6137375828
N                 0.1767218070        1.1161878405       -0.0051280785
C                -1.0921177328        1.4614369929        0.4636089473
C                -1.3757850553        2.6947885168        1.0916352093
N                -2.5660826651        2.9967331546        1.5093262348
C                -3.5844912719        2.1285329380        1.3566972838
C                -4.8389104824        2.5450963505        1.8406969676
C                -5.9108770371        1.7141002103        1.7176118981
C                -5.7651408276        0.4565291088        1.1155022875
C                -4.5687329200       -0.0093912236        0.6231016510
C                -4.4979871052       -1.3951197485       -0.0178893920
C                -3.5460799708       -2.3014641568        0.7804295971
C                -5.8548222916       -2.1172190324       -0.0197870698
C                -4.0884495351       -1.2763894968       -1.4954562410
C                -3.4291906333        0.8476836079        0.7405854390
C                -2.1207253540        0.5592518581        0.2985735859
C                 2.4144356935        0.3721236422       -1.7215855337
S                 3.9763734653       -0.1624894380       -2.1857594983
H                 4.8273646003       -2.5899135019        1.7390479230
H                 4.2987788915       -1.0706556907        1.0262299334
H                 4.6384322147       -2.4624429467       -0.0106583039
H                 7.0393894310       -2.3831054448        0.7579806118
H                 6.6776354241       -0.9041680029        1.6467183473
H                 7.7954307699       -0.5565701677       -0.5674351590
H                 6.3875001798       -1.2319680031       -1.3804120633
H                 6.2163093428        1.1923175383        0.4684461298
H                 7.6865344815        1.5707632546       -1.4633993359
H                 6.2884642107        2.4077143798       -1.6510624671
H                 0.2339383912        0.2028979353       -0.4363211808
H                -0.5998016024        3.4290941532        1.2439832544
H                -4.9066573358        3.5191894732        2.2973175895
H                -6.8841047762        2.0101716851        2.0805843274
H                -6.6522574565       -0.1497599596        1.0511808224
H                -2.5456115222       -1.8973416801        0.8787620200
H                -3.4754411844       -3.2763843065        0.3015461354
H                -3.9408846544       -2.4424405652        1.7847781017
H                -5.7322254760       -3.0927200491       -0.4867996233
H                -6.5990458473       -1.5684012120       -0.5917246693
H                -6.2230239328       -2.2784904531        0.9904397282
H                -3.1465905758       -0.7612179767       -1.6421027419
H                -4.8516936114       -0.7212259642       -2.0374428104
H                -4.0085802747       -2.2677042644       -1.9378180051
H                -1.8854209900       -0.3766764613       -0.1833965020
H                 1.5625345888        0.1281263717       -2.3251506712



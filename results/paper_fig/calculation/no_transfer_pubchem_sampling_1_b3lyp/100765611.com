%chk=./calculation/no_transfer_pubchem_sampling_1_b3lyp/100765611.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 3.3495116680       -1.5286677533        0.8751958924
C                 2.7477249592       -0.1981294070        0.5482389928
C                 3.3134987283        0.9784064926        1.0278601559
C                 2.7683278498        2.2073037197        0.7046215752
C                 1.6422056832        2.2809550372       -0.0983643695
C                 1.0669080947        1.1164516797       -0.5893802603
C                -0.1761572695        1.1610180111       -1.4441979599
N                -1.2213785800        0.3517094801       -0.8685849839
N                -1.7167630621       -0.6969884109       -1.5319705382
C                -2.6230659393       -1.2369557361       -0.7467314422
C                -3.4002293398       -2.4422623021       -1.1424149794
F                -4.7255515605       -2.2144107350       -1.1393542599
F                -3.0879706659       -2.8849552575       -2.3629689055
F                -3.1964967072       -3.4705884576       -0.2963134599
C                -2.7261816363       -0.5308517920        0.4766462704
C                -3.5482884153       -0.7869466934        1.5772470422
N                -4.2017601101       -0.9693505438        2.5023085302
C                -1.7823863886        0.4970485493        0.3506577209
N                -1.4929916589        1.5262316362        1.1913973545
C                 1.6238336273       -0.1127427206       -0.2615968590
H                 3.8548558413       -1.5013489632        1.8375001056
H                 4.0844926305       -1.7972483653        0.1160449605
H                 2.5890642441       -2.3055464882        0.8948821362
H                 4.1888749300        0.9290127906        1.6583455260
H                 3.2174757987        3.1133988125        1.0825317945
H                 1.2180417442        3.2426013523       -0.3510914648
H                -0.5320055380        2.1903674857       -1.5577761857
H                 0.0170974933        0.7400781405       -2.4339893249
H                -1.8534786125        1.4328379917        2.1278728571
H                -0.5687661093        1.9254905183        1.1217588985
H                 1.1715593005       -1.0159160715       -0.6466348199



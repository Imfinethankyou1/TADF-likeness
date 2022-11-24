%chk=./calculation/no_transfer_pubchem_sampling_0_b3lyp/55538781.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C               -10.1549032919       -1.0414937883       -0.8041954872
O                -8.8116746150       -1.1312650244       -0.3561405107
C                -7.8964654240       -0.5641532433       -1.1676699596
O                -8.1828554889       -0.0149418927       -2.2055480693
C                -6.5447344911       -0.6911898047       -0.6510555084
S                -5.2390624251       -0.0088103115       -1.5811902920
C                -4.1439649638       -0.5848640621       -0.3221361468
N                -2.8118721285       -0.3493533531       -0.4534170379
C                -1.8144952750       -0.7350570855        0.4491398963
O                -2.0529579656       -1.3354665360        1.4670207160
C                -0.4783841609       -0.3242841547        0.0062442437
C                 0.5919670506       -0.6122725160        0.7647084093
C                 1.9742993364       -0.2950376248        0.5009635634
C                 2.9449219820       -0.6679410187        1.4359251981
C                 4.2800387388       -0.3859525114        1.2433517207
C                 4.6955070594        0.2817820930        0.0891889407
N                 6.0213932194        0.5674302684       -0.1685348313
C                 7.0990340049        0.2152076885        0.7204709109
C                 8.3794339720        0.6487704049        0.0070102774
C                 8.1243345741        2.0565113266       -0.4811661215
S                 6.4350458928        1.9034514132       -1.1377796895
O                 5.6287807963        3.0328640472       -0.7914842131
O                 6.4646409573        1.5071241347       -2.5104747218
C                 3.7369303548        0.6487677121       -0.8601499734
C                 2.4065218860        0.3708689098       -0.6534369246
N                -4.7201131290       -1.2020439099        0.6481863181
C                -6.0584192103       -1.2796548547        0.4974685411
C                -6.8537365171       -1.9673220809        1.5462561350
H               -10.7586047014       -1.5388574928       -0.0493827532
H               -10.4528596493        0.0041113875       -0.9077478725
H               -10.2671622517       -1.5338092571       -1.7726074153
H                -2.5354537927        0.1543017341       -1.2843475083
H                -0.3959624473        0.2062012498       -0.9300228200
H                 0.3950125945       -1.1458312364        1.6874637634
H                 2.6385852342       -1.1836448667        2.3344234492
H                 4.9983508473       -0.6839105410        1.9907674590
H                 7.0973860534       -0.8639608553        0.8966458003
H                 7.0066997825        0.7344891470        1.6859526255
H                 8.5447753785       -0.0011220640       -0.8536848295
H                 9.2392216456        0.5846725472        0.6735501835
H                 8.0932921939        2.7891576806        0.3198909235
H                 8.7831153178        2.3776771326       -1.2798244372
H                 4.0619619349        1.1538641444       -1.7550935405
H                 1.6904383857        0.6689899195       -1.4024703780
H                -7.5829989037       -1.2849734314        1.9778973009
H                -7.3993665369       -2.8064183355        1.1197195758
H                -6.1756428239       -2.3203250879        2.3173170899



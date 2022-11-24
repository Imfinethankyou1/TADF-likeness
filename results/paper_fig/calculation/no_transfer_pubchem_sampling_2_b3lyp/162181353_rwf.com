%chk=calculation/no_transfer_pubchem_sampling_2_b3lyp/162181353_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_2_b3lyp/162181353_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 8.0104547724        2.1422194292        2.0965237953
S                 6.9305046751        2.1204838521        0.6287097342
C                 5.8006376646        0.7814666158        0.9597345447
C                 5.8504352874       -0.0374941680        2.0950729557
C                 4.9175614741       -1.0656618758        2.2421183008
C                 3.9330788163       -1.2992222218        1.2864564108
C                 3.8704677404       -0.4667502933        0.1591133673
N                 2.9236023950       -0.6650352454       -0.8726186248
C                 1.6268458699       -1.0836176829       -0.8002716505
C                 0.8777388589       -1.3315005249        0.3097580010
C                -0.5004518313       -1.7729691270        0.2808615864
O                -1.1188676314       -2.0049123055        1.3290554649
C                -1.2473480711       -1.9927481417       -1.0319705299
N                -0.6686486893       -2.7334015014       -1.9059728056
C                -2.6326047415       -1.4149726345       -1.1810870801
C                -3.0926963711       -1.1592782127       -2.4321795861
C                -4.4970787099       -0.7904826745       -2.8446016892
N                -4.6723457259        0.4950786956       -3.5427380406
N                -3.3649219051       -1.2135592071       -0.0069456669
C                -4.0214520014       -0.0184768209        0.3623837286
C                -4.9041774016       -0.0646983512        1.4541658982
C                -5.5481358737        1.0892120988        1.8940572048
C                -5.3374740476        2.3108404018        1.2505863682
C                -4.4570214068        2.3621733951        0.1687908179
C                -3.7970437202        1.2144663074       -0.2704151503
C                 4.8022987530        0.5635675045        0.0007487968
H                 8.5805241110        1.2143326763        2.1925620323
H                 7.4417030957        2.3330206047        3.0104605809
H                 8.7069749830        2.9686779496        1.9350059642
H                 6.6057088479        0.1095932292        2.8580900377
H                 4.9705912052       -1.7090750418        3.1161274806
H                 3.2470848857       -2.1299595627        1.3947485969
H                 3.2380395017       -0.3903343849       -1.7927847021
H                 1.1793466800       -1.2350754295       -1.7764500973
H                 1.2712208298       -1.1891140442        1.3096998192
H                -1.3174337358       -2.9268036399       -2.6791924885
H                -2.3797845872       -1.2515802531       -3.2507534970
H                -5.1695492547       -0.7986284266       -1.9839782238
H                -4.8566905465       -1.5751230538       -3.5265691134
H                -4.4631308582        1.2555471400       -2.8988297129
H                -3.9979670491        0.5691659285       -4.3039958768
H                -2.9687025986       -1.7153451687        0.7844160324
H                -5.0826759510       -1.0166964198        1.9474828654
H                -6.2271109486        1.0291344571        2.7408083884
H                -5.8491512883        3.2075140169        1.5878033100
H                -4.2659018908        3.3083633436       -0.3319033286
H                -3.0877741427        1.2725555210       -1.0896969580
H                 4.7413175325        1.2095442462       -0.8723852612



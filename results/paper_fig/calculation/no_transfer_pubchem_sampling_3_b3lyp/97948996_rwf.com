%chk=calculation/no_transfer_pubchem_sampling_3_b3lyp/97948996_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_3_b3lyp/97948996_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 9.3406441785        0.7075233414       -3.2382677722
C                 9.2875531321        0.7585665659       -1.7183337646
O                 7.9747840590        0.4104689354       -1.2237104735
C                 7.0653786893        1.4141349704       -1.1624732890
O                 7.3095441934        2.5603725912       -1.4933206712
C                 5.7507599635        0.9549139034       -0.6448234922
C                 4.7211281847        1.8997377938       -0.5254158906
C                 3.4750382858        1.5249148189       -0.0481884537
C                 3.2268311918        0.1899949515        0.3226626919
N                 1.9354478118       -0.1052018570        0.7970331231
C                 1.4254232785       -1.3156805435        1.1897519764
O                 2.0416513611       -2.3731785659        1.1965537495
C                -0.0398816622       -1.2729964791        1.6567951926
C                -1.0370471116       -0.6595861743        0.6399693585
C                -1.2187693593        0.8440268758        0.8272799847
O                -0.3322955874        1.6842480510        0.8687911228
N                -2.5684971299        1.0696438940        0.9825269935
C                -3.1225947141        2.3747458390        1.2077095436
C                -3.8879809433        2.9894017374        0.2211873238
C                -4.4100497419        4.2659272890        0.4299003837
C                -4.1518784466        4.9327307885        1.6349161920
O                -4.6062446967        6.1775430924        1.9423288090
C                -5.4060087492        6.8614370784        0.9877825346
C                -3.3726575510        4.3105222476        2.6231917618
C                -2.8643437460        3.0376092549        2.4114826403
C                -3.3077712771       -0.1385117421        0.9388709182
S                -4.9564122672       -0.2576195695        1.0593024971
N                -2.4077459124       -1.1467206747        0.7832904873
C                -2.7892967431       -2.5245744183        0.4807477330
C                -3.0251840725       -2.7721824590       -1.0247491594
C                -3.4287526730       -4.2066254928       -1.2898404418
C                -2.4732771956       -5.1732403425       -1.6293382380
C                -2.8437443775       -6.5029973879       -1.8356650056
C                -4.1794142669       -6.8855924089       -1.7029421636
C                -5.1407591230       -5.9312031601       -1.3638207802
C                -4.7671129001       -4.6029098378       -1.1586531583
C                 4.2525208736       -0.7633134039        0.2062925654
C                 5.4988345721       -0.3727668424       -0.2742077870
H                10.3638031132        0.8998508944       -3.5817383827
H                 8.6843609589        1.4679053100       -3.6704864555
H                 9.0335199121       -0.2771269181       -3.6059956907
H                 9.5518640638        1.7517749803       -1.3473754018
H                 9.9509186711        0.0145343216       -1.2697511504
H                 4.9193165195        2.9268969137       -0.8130428018
H                 2.6806567017        2.2620514344        0.0407230558
H                 1.2939483356        0.6894490191        0.8440264486
H                -0.3070386933       -2.3141208648        1.8442899971
H                -0.1126081463       -0.7390101778        2.6122545651
H                -0.6771567156       -0.8456834905       -0.3829495839
H                -4.0863870547        2.4690582256       -0.7097681394
H                -5.0093940356        4.7252519089       -0.3469441774
H                -6.3306716488        6.3115005392        0.7703536898
H                -4.8572466089        7.0334719317        0.0528114267
H                -5.6550486156        7.8218023879        1.4419787674
H                -3.1835077499        4.8445092532        3.5484613917
H                -2.2651421850        2.5553846888        3.1771865598
H                -2.0005126920       -3.1853115124        0.8525033242
H                -3.7000980599       -2.7405862144        1.0426529209
H                -3.8083188738       -2.0855155429       -1.3647492878
H                -2.1111598371       -2.5309570618       -1.5822094019
H                -1.4307660715       -4.8813547423       -1.7397813568
H                -2.0893273355       -7.2380489802       -2.1034260708
H                -4.4701278310       -7.9198727229       -1.8659524986
H                -6.1836365935       -6.2199340651       -1.2626546947
H                -5.5200470718       -3.8621919704       -0.8978835446
H                 4.0623668841       -1.7880402116        0.4914913123
H                 6.2896211321       -1.1092449932       -0.3630548618



%chk=./calculation/no_transfer_pubchem_sampling_6_b3lyp/17003127.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 6.0615754600        3.3975335975       -1.0272551468
C                 6.5478273752        1.9712071061       -0.7990713690
C                 5.5294663747        0.9110024725       -1.2217805262
C                 4.2023706441        0.9464499973       -0.4616808424
C                 4.3597405802        0.6919818612        1.0373714981
C                 3.0175125427        0.5299135554        1.7650395754
C                 2.6143020770       -0.9244852935        2.0175885641
C                 2.3380944492       -1.7195232900        0.7424351111
C                 1.7317355001       -3.0956970551        1.0236385805
C                 0.2415783418       -3.0413546276        1.3865954312
N                -0.5594441990       -2.5933773918        0.2781223940
C                -1.1255134681       -1.3642046675        0.0614421549
C                -1.1115043239       -0.2691852607        1.0778084301
C                -2.1120787937       -0.5688268724        2.2041691885
C                -3.3806219693        0.1770284164        1.7927215947
O                -4.4833902127        0.0088923038        2.2612987684
N                -3.0360656616        1.0730568599        0.8484657693
C                -3.9587563373        1.9171000027        0.1726321138
C                -3.9873032349        3.2798928775        0.4667390493
C                -3.0845152657        3.8662309678        1.5109530999
C                -4.8810046549        4.0896939016       -0.2208792896
C                -5.7322592761        3.5541585385       -1.1689003801
C                -5.7063705778        2.1977224690       -1.4350024095
C                -4.8226728153        1.3580593298       -0.7712779668
C                -4.8009432468       -0.1115592585       -1.0433582113
C                -1.6389474454        1.0435683515        0.4766947623
N                -1.7215090162       -1.2722390064       -1.0935023290
C                -1.5637166258       -2.4905721623       -1.7015933210
C                -2.0093388867       -2.9452157951       -2.9365504687
C                -1.7059595180       -4.2444821244       -3.2837782008
C                -0.9832429234       -5.0810068077       -2.4311201782
C                -0.5325653409       -4.6474672616       -1.2003503345
C                -0.8275291632       -3.3403981210       -0.8464002439
H                 5.7519518031        3.5328418994       -2.0614507542
H                 6.8585071899        4.1060719118       -0.8127697815
H                 5.2192567704        3.6326475355       -0.3818382343
H                 7.4601215124        1.8131788510       -1.3796921909
H                 6.8064661642        1.8377253649        0.2526607265
H                 5.9819868997       -0.0755728677       -1.0902592156
H                 5.3193619778        1.0358488385       -2.2870982259
H                 3.5479890987        0.1874129841       -0.8934855491
H                 3.7182572802        1.9128802504       -0.6126657041
H                 4.8995611875        1.5316384031        1.4772675302
H                 4.9680517444       -0.2016108400        1.1925617676
H                 3.0750016330        1.0276989381        2.7359864954
H                 2.2336457471        1.0297960297        1.1902419940
H                 1.7239208782       -0.9263335996        2.6521683362
H                 3.4077291486       -1.4215100141        2.5811044349
H                 3.2760801620       -1.8599359943        0.2039374812
H                 1.6651287219       -1.1580929640        0.0893728268
H                 2.2713030808       -3.5739048666        1.8432663791
H                 1.8439596308       -3.7209358246        0.1358636893
H                -0.0971372714       -4.0439681349        1.6693338533
H                 0.0775505290       -2.3766109685        2.2355543444
H                -0.0979048400       -0.1389412183        1.4621000107
H                -1.7735019457       -0.1732721801        3.1642034236
H                -2.3308972654       -1.6287468584        2.3271295184
H                -3.5044029379        4.7922944794        1.8959494592
H                -2.9534246198        3.1714968982        2.3380781191
H                -2.1042551501        4.0918527095        1.0888718403
H                -4.9130187781        5.1472116906       -0.0016258399
H                -6.4244424604        4.1949784290       -1.6950925090
H                -6.3813428615        1.7803844560       -2.1680001411
H                -3.7833361361       -0.4602265227       -1.2219343743
H                -5.4189929401       -0.3517548898       -1.9045794761
H                -5.1895773651       -0.6473814129       -0.1775138898
H                -1.5386985447        1.0527860158       -0.6106519839
H                -1.1063959992        1.9056366729        0.8963145435
H                -2.5742712556       -2.2979028597       -3.5877629999
H                -2.0368840491       -4.6320011364       -4.2358908060
H                -0.7741206463       -6.0937484447       -2.7424574717
H                 0.0178215183       -5.3058273731       -0.5445394945



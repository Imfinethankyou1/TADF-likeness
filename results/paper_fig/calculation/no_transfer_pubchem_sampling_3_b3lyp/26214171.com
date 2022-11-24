%chk=./calculation/no_transfer_pubchem_sampling_3_b3lyp/26214171.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 8.2679614908        2.3666886918        1.9432668188
O                 7.1444785905        2.9350328348        1.3164733470
C                 5.9670906426        2.2507947067        1.3451546781
C                 4.8875188115        2.8702038226        0.7135602624
C                 3.6516242051        2.2619350506        0.6868811465
C                 3.4511194306        1.0241136204        1.2889570734
C                 2.1133136105        0.3463781702        1.2219891567
C                 2.0284557908       -0.5334649005       -0.0367389294
N                 0.7517562921       -1.1787829273       -0.1416986404
N                 0.5858186286       -2.4174142169        0.3124479094
C                -0.6992706752       -2.7293278134        0.1358646853
C                -1.3742597463       -4.0056137639        0.4426733901
O                -2.5136287779       -4.1831676736        0.0238475888
N                -0.7492232311       -4.9537273042        1.1813197430
C                -1.4533167445       -6.1824175789        1.4655814251
C                 0.5704214793       -4.8786441986        1.7492467816
C                -1.3713738486       -1.6344858466       -0.4523948743
C                -0.4076180456       -0.6633317608       -0.6069789725
C                -0.6607703724        0.6686180194       -1.2068573947
C                -1.9900551792        0.5654044509       -1.9650948568
N                -3.0064344308       -0.0067890878       -1.1006209112
C                -4.3533552256        0.2456583390       -1.5793485361
C                -4.6718199933        1.7027739074       -1.5458858652
C                -4.4183478175        2.6092719605       -0.5015936120
N                -4.8343829709        3.8212559399       -0.8129077470
N                -5.3562342211        3.7323198162       -2.0313326407
C                -5.9191849276        4.8842565573       -2.6725076448
C                -5.2763779436        2.4748156920       -2.5155362506
C                -2.7814446062       -1.4257258676       -0.8690508391
C                 4.5283366835        0.4130699262        1.9106982911
C                 5.7763297494        1.0121756525        1.9463415617
H                 8.1007174397        2.2227035036        3.0157082718
H                 9.0781614244        3.0797021974        1.7989568703
H                 8.5432991150        1.4104127534        1.4864239350
H                 5.0473137237        3.8320219809        0.2526487778
H                 2.8260059165        2.7579962939        0.1972474952
H                 1.3156944968        1.0905079836        1.1888759438
H                 1.9605219619       -0.2853579813        2.0985882202
H                 2.1966553083        0.0790440558       -0.9270568019
H                 2.7896135060       -1.3155208975        0.0083069918
H                -0.9022362120       -7.0360566224        1.0630489248
H                -1.5672588741       -6.3125230080        2.5447682564
H                -2.4350375518       -6.1302336546        0.9993084543
H                 1.0315298929       -3.9316493731        1.4869362207
H                 0.5106204129       -4.9758986109        2.8377463300
H                 1.1853597503       -5.6976292270        1.3647631679
H                 0.1434843864        0.9682478705       -1.8837783368
H                -0.7525159383        1.4268159505       -0.4240674303
H                -2.3157449025        1.5626701445       -2.2680306510
H                -1.8435079905       -0.0428638587       -2.8770740305
H                -5.0385411552       -0.2885681836       -0.9120036817
H                -4.5115645770       -0.1429426249       -2.6034376974
H                -3.9499088609        2.4138559048        0.4410742736
H                -5.4037080495        5.0953220924       -3.6118301572
H                -6.9810711543        4.7327390938       -2.8775422415
H                -5.7995057743        5.7283912857       -1.9966713854
H                -5.6500678851        2.2195858016       -3.4871793508
H                -3.4600985325       -1.7747033381       -0.0868945786
H                -2.9868158224       -2.0332371371       -1.7706483788
H                 4.3948603748       -0.5490189401        2.3841839764
H                 6.5866239230        0.5043113269        2.4452084677



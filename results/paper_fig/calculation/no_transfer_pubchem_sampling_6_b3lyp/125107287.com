%chk=./calculation/no_transfer_pubchem_sampling_6_b3lyp/125107287.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -1.3498336392       -4.6152664201       -2.1496184889
C                -2.1341222435       -4.7389714371       -0.8463528873
C                -2.8565855048       -3.4524663214       -0.4178658730
C                -3.6333713653       -2.7975534531       -1.5563663895
N                -1.9181302675       -2.4934856459        0.1371598380
C                -1.9533198774       -2.1272249239        1.4249193410
O                -2.7326066035       -2.5906498313        2.2369232425
C                -0.8775238603       -1.1058963069        1.8415140613
C                -1.2159754014       -0.4297974083        3.1874288607
C                -2.2512713686        0.6610656156        3.1593907709
C                -3.5748003849        0.3944328408        2.8223323803
C                -4.5127785024        1.4101545241        2.7951875406
C                -4.1507150660        2.7052511987        3.1259916286
C                -2.8478948834        2.9734319398        3.5024570661
C                -1.9075420554        1.9581928653        3.5162594442
N                -0.5335426477       -0.1790322869        0.7737530403
C                -1.6080584741        0.5593965749        0.1442353395
C                -1.5660888263        2.0557107336        0.3454283849
C                -0.5303712400        2.7073662233        0.9939934242
C                -0.5559892571        4.0817520778        1.1604023600
C                -1.6204818970        4.8376279710        0.6955984653
C                -1.6686148877        6.3224984254        0.8873019907
C                -2.6614826009        4.1797295262        0.0489565891
C                -2.6338088170        2.8116213117       -0.1255087218
C                 0.7358506263       -0.1788218569        0.3290182635
O                 1.6062613610       -0.8747387405        0.8405363031
C                 1.1134203023        0.7039302555       -0.8582304192
N                 1.9603058619       -0.0268383368       -1.7788874496
C                 3.3508370975        0.1689997335       -1.8030094763
C                 4.0071450482        0.0290327539       -3.0211098141
C                 5.3836801890        0.2028489668       -3.1283191344
C                 6.0678620815        0.5409694661       -1.9855583175
C                 5.4104655836        0.6821767074       -0.7624164210
C                 4.0570280346        0.4932348011       -0.6365229017
O                 6.3193745683        1.0070410745        0.2077688786
C                 7.5723029787        1.0692839294       -0.4405198152
O                 7.4072538919        0.7766722159       -1.8113619618
S                 1.3882272360       -1.5364168963       -2.2235632051
O                 1.6321900008       -1.7356420446       -3.6193289021
O                 0.0016386965       -1.5381253111       -1.8133377421
C                 2.2691415418       -2.8238913423       -1.2843548661
C                 3.5729727836       -3.1922864860       -1.9514890035
H                -0.7030668889       -5.4808054613       -2.2775859668
H                -2.0226373706       -4.5727441896       -3.0018150381
H                -0.7362300697       -3.7184740809       -2.1578957060
H                -2.8766251401       -5.5330778179       -0.9487277040
H                -1.4594612904       -5.0237500257       -0.0356989732
H                -3.5404418934       -3.7019154703        0.4018843512
H                -2.9451288345       -2.3992453996       -2.2973104847
H                -4.2888236122       -3.5228555386       -2.0306193581
H                -4.2402728912       -1.9807355829       -1.1721205520
H                -1.2277540393       -2.1145053799       -0.5001702096
H                 0.0303969925       -1.7011516139        2.0154217951
H                -0.2912757309       -0.0295794523        3.6074039753
H                -1.5722274064       -1.2348543267        3.8358074108
H                -3.8708761564       -0.6188000629        2.5971252884
H                -5.5348156717        1.1880877586        2.5238036962
H                -4.8848203153        3.4971877357        3.1046581138
H                -2.5591580077        3.9756979543        3.7830442298
H                -0.8936197343        2.1780140864        3.8189515308
H                -1.6278158930        0.3303845057       -0.9295013010
H                -2.5446655003        0.1880623611        0.5668070264
H                 0.2978671243        2.1424630777        1.3935549019
H                 0.2624250680        4.5688749101        1.6712393730
H                -2.5349970930        6.5965580993        1.4880945942
H                -1.7587119131        6.8279383375       -0.0732675618
H                -0.7724324340        6.6813887203        1.3867325928
H                -3.5037082153        4.7495406691       -0.3159729598
H                -3.4533479859        2.3161091010       -0.6256135323
H                 1.6616287674        1.5805653542       -0.5086410663
H                 0.2405046634        1.0404733685       -1.4159415686
H                 3.4266963403       -0.2397550133       -3.8873778856
H                 5.8925539570        0.0893143013       -4.0700268943
H                 3.5703901446        0.5584713024        0.3213313795
H                 7.9961003422        2.0800988595       -0.3274026124
H                 8.2583752415        0.3363824972        0.0135483380
H                 1.5697628768       -3.6567086512       -1.2740575881
H                 2.3917882809       -2.4324360798       -0.2784888456
H                 3.4074603203       -3.3292387173       -3.0175296998
H                 4.3161553711       -2.4114566937       -1.8144981140
H                 3.9597613852       -4.1188381250       -1.5319773984



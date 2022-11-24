%chk=./calculation/no_transfer_pubchem_sampling_1_b3lyp/98008575.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 5.6819068835        0.9598226275        0.7048408738
C                 4.6664841059        1.4317367119       -0.3381746554
C                 5.3047067658        1.4954416419       -1.7272799118
C                 3.4527278878        0.5358713945       -0.3444784661
C                 2.2003133671        1.0383717392       -0.0201775163
C                 1.0794462004        0.2272578607       -0.0129120923
C                 1.1816446466       -1.1183929132       -0.3376808463
C                 0.0007495374       -2.0556699321       -0.2924310008
C                -1.3366945618       -1.3920142964       -0.6243910390
N                -1.8250899038       -0.9115073960        0.6504557602
C                -2.8689276013       -0.0737375811        0.8231842718
O                -3.2508048407        0.2891646775        1.9129739592
O                -3.3869326279        0.2787784669       -0.3704273355
C                -4.5057831403        1.1930177410       -0.4355182261
C                -4.7803822995        1.3349419766       -1.9336979753
C                -4.1241902689        2.5444194404        0.1686314555
C                -5.7207548786        0.5988305023        0.2765995400
C                -1.1032471326       -1.4702462128        1.7778585939
C                -0.2461057477       -2.5682257029        1.1546708549
C                 1.0660040560       -2.8600239016        1.8434834840
O                 1.6700247960       -3.8912365141        1.7161442558
O                 1.5168249324       -1.8533403261        2.5944041112
C                 2.4337819296       -1.6242405721       -0.6689284597
C                 3.5507954153       -0.8116389848       -0.6705427102
H                 6.5297478109        1.6404596264        0.7393869610
H                 6.0500391860       -0.0343262695        0.4637815084
H                 5.2238858310        0.9273376792        1.6911867231
H                 4.3425941150        2.4423937425       -0.0652510188
H                 4.5758084241        1.8261911083       -2.4639966547
H                 5.6768625085        0.5192404431       -2.0276474037
H                 6.1381584560        2.1945159604       -1.7263890610
H                 2.0965869151        2.0833201379        0.2366351933
H                 0.1246587892        0.6562767510        0.2482663734
H                 0.1989913890       -2.8993859821       -0.9569386672
H                -2.0390554020       -2.1200769997       -1.0488352659
H                -1.2303010157       -0.5685842459       -1.3338752872
H                -5.0149656915        0.3643493961       -2.3626957989
H                -3.9032087900        1.7317758810       -2.4380643608
H                -5.6177210588        2.0070651083       -2.0953642521
H                -4.9424514560        3.2474736919        0.0406677583
H                -3.2388469652        2.9386551640       -0.3263128361
H                -3.9152084924        2.4325340715        1.2289101773
H                -6.5767429825        1.2560687064        0.1519021571
H                -5.5119478413        0.4850730359        1.3366555717
H                -5.9593636277       -0.3764584829       -0.1426894961
H                -1.8048114215       -1.8506501496        2.5241784357
H                -0.4812558599       -0.7040930728        2.2516064488
H                -0.8020266291       -3.5081264032        1.0964131636
H                 2.3802899180       -2.0799392990        2.9750157363
H                 2.5352864905       -2.6707816295       -0.9196884813
H                 4.5084998796       -1.2376904170       -0.9297605494



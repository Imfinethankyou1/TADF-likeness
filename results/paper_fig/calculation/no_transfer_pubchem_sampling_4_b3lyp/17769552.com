%chk=./calculation/no_transfer_pubchem_sampling_4_b3lyp/17769552.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -3.0414950005        3.4373761281       -0.7001197096
C                -3.8368626996        2.1822783555       -1.0496054973
C                -2.9931874350        0.9026546065       -1.0981877722
C                -2.8050652768        0.1222806729        0.2370190568
C                -4.0571491552       -0.7559806670        0.3934538049
C                -2.7533589233        0.9711109491        1.5240123021
C                -1.5225342420       -0.7114955468        0.1494033895
C                -1.5652212241       -2.0731978748        0.2138584297
C                -0.4028320432       -2.8602001073        0.1572796222
C                -0.5191446261       -4.2641810772        0.2285665266
C                 0.5818665122       -5.0673398060        0.1813602769
C                 1.8431885677       -4.4726791630        0.0600289098
C                 1.9817756260       -3.1136832961       -0.0109580370
C                 0.8659280086       -2.2378971022        0.0338688431
C                 0.9406588272       -0.8089906496       -0.0374232792
C                 2.3354778362       -0.1720807235       -0.1700776230
C                 3.1477647927       -0.5507444580        1.0921340526
C                 2.9633855543       -0.6771079172       -1.4924424884
C                 2.3634203915        1.3704265796       -0.2466849441
C                 3.7673683983        1.9676240905       -0.3708681083
C                 3.6929731283        3.4886354902       -0.4548271783
C                -0.2429380446       -0.0867617957        0.0195717375
O                -0.1911747151        1.2751752008       -0.0489391822
H                -2.6619563725        3.4110136199        0.3173313238
H                -3.6763977100        4.3160522117       -0.7935012379
H                -2.2006800227        3.5575321705       -1.3809685046
H                -4.6669360630        2.0649865837       -0.3513154572
H                -4.2743612430        2.3297261065       -2.0400548917
H                -2.0395961324        1.1425845864       -1.5724289741
H                -3.4798262558        0.2033186664       -1.7830079444
H                -4.1730313724       -1.4422579030       -0.4418077446
H                -4.0298574139       -1.3253682748        1.3193112035
H                -4.9366723448       -0.1158252963        0.4219396532
H                -3.6535743838        1.5723026863        1.6244730205
H                -2.7119528932        0.2911506332        2.3740144465
H                -1.8912412399        1.6261599173        1.6074218758
H                -2.4997624271       -2.5986014162        0.3144670588
H                -1.5085080563       -4.6902158578        0.3221321353
H                 0.4913445042       -6.1421321730        0.2360857911
H                 2.7232040858       -5.0989859823        0.0217929308
H                 2.9808928956       -2.7287215739       -0.1036665709
H                 2.8002993148       -1.4685167641        1.5542979504
H                 4.2054607381       -0.6578911313        0.8628299892
H                 3.0410063049        0.2347845457        1.8377294080
H                 4.0394701484       -0.8023503563       -1.3964158787
H                 2.5352631715       -1.6161977760       -1.8267403305
H                 2.7773748839        0.0491671153       -2.2812724915
H                 1.7818038327        1.7031023214       -1.1063218142
H                 1.9011546285        1.7886979552        0.6470760723
H                 4.2651855726        1.5909606439       -1.2648350482
H                 4.3755685332        1.6937962350        0.4915953638
H                 3.2185302730        3.8982347156        0.4345270295
H                 3.1112632280        3.7964132441       -1.3213130125
H                 4.6890227869        3.9181391333       -0.5407233068
H                -1.0853322290        1.6337175249       -0.0630521767



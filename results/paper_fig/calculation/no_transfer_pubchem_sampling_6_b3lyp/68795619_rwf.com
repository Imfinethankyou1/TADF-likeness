%chk=calculation/no_transfer_pubchem_sampling_6_b3lyp/68795619_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_6_b3lyp/68795619_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 0.5742428903        1.8210225599        1.8820913888
C                -0.6240560690        1.8586438878        1.2694768452
C                -0.9406958018        0.8358584097        0.2788436945
N                -2.2115302126        0.7307247155       -0.2618699026
C                -3.4500108682        1.0497313941        0.3188332815
C                -4.5511093224        1.2316608359       -0.5367603501
C                -5.8095173906        1.5237389818       -0.0195939898
C                -5.9966747244        1.6590145003        1.3584880720
C                -4.9076084417        1.4756566459        2.2108074166
C                -3.6461344463        1.1595534789        1.7056727243
C                 0.0611384010        0.0149384379       -0.1509817467
N                -0.1697220046       -0.9956440216       -1.1293799081
C                -0.3257539600       -2.3547659758       -0.7676108229
C                -0.7493192341       -2.7344486259        0.5146667476
C                -0.9410374565       -4.0824199136        0.8166376051
C                -0.7175578696       -5.0723325373       -0.1415164048
C                -0.2947048154       -4.6942462078       -1.4181115013
C                -0.0996091914       -3.3515739441       -1.7313581640
C                 1.4807402944        0.2761416505        0.2797870427
N                 2.1789270742        1.0221431307       -0.7252773277
C                 3.5705498721        1.2227797145       -0.7396192246
C                 4.4364957541        0.6118564699        0.1818704150
C                 5.8152069607        0.7988719549        0.0757430619
C                 6.3577898812        1.6049106654       -0.9248437169
C                 5.4963314738        2.2291961644       -1.8303676316
C                 4.1206682024        2.0383208231       -1.7443090653
O                 1.5397570513        0.9210880172        1.5988840019
H                 0.8645402361        2.4971842267        2.6808953907
H                -1.3474110685        2.6182233926        1.5335695462
H                -2.2416908600        0.1189666506       -1.0703059339
H                -4.4056564179        1.1498768992       -1.6113456828
H                -6.6467002652        1.6570047627       -0.6998662109
H                -6.9772597025        1.8964761202        1.7604237464
H                -5.0392202058        1.5576435230        3.2867611455
H                -2.8214772959        0.9754175178        2.3851949206
H                 0.4147366599       -0.8880834272       -1.9538093666
H                -0.9230812965       -1.9715402591        1.2666286786
H                -1.2682797432       -4.3578273177        1.8161929064
H                -0.8684032944       -6.1200796644        0.1014243149
H                -0.1108582906       -5.4494640659       -2.1782183521
H                 0.2293323934       -3.0664836157       -2.7291283054
H                 2.0140870037       -0.6633893668        0.4417821164
H                 1.6358622851        1.7800787217       -1.1224646701
H                 4.0399253815        0.0175994393        0.9974363850
H                 6.4680975219        0.3171328498        0.7992159900
H                 7.4316983810        1.7509542786       -0.9944565262
H                 5.8966619668        2.8659597616       -2.6152384419
H                 3.4582905641        2.5139303608       -2.4648931914



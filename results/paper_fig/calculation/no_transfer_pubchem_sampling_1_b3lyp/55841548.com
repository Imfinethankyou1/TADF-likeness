%chk=./calculation/no_transfer_pubchem_sampling_1_b3lyp/55841548.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 9.0886363775        2.6635269562        0.3305082844
C                 7.9618487506        1.7887456783        0.7815568093
C                 7.7296561955        1.5056466772        2.1198698038
C                 6.6856225735        0.6853872071        2.5125051867
C                 5.8430882026        0.1120379989        1.5717913250
C                 4.6938622128       -0.7737840844        1.9743270280
N                 3.4395489942       -0.3009340391        1.4282104994
C                 2.7716967516       -0.9662489598        0.4564786916
O                 3.1424022607       -2.0398448236        0.0306663918
C                 1.4880051327       -0.2743519672        0.0095350188
C                 1.1380924655       -0.6821506012       -1.4275617670
C                -0.3754283554       -0.7442843612       -1.6157085054
N                -1.0111220540       -1.7358635634       -0.7500886588
C                -1.4549185032       -2.9238068551       -1.4388734837
C                -2.7380273115       -2.6592704156       -2.2300894144
O                -3.0983135713       -3.3767915032       -3.1385559309
N                -3.3957815449       -1.5714292258       -1.7545371701
C                -4.5607766107       -0.9773100000       -2.1991393067
C                -5.2549509890       -1.4563984172       -3.3122524947
C                -6.4013381278       -0.8280642344       -3.7511618634
C                -6.8805096359        0.2956348727       -3.0947356274
C                -6.2183422998        0.7649363967       -1.9779309091
C                -5.0696270785        0.1366531439       -1.4997320606
C                -4.3818797287        0.6489675095       -0.2850740559
O                -3.1706739834        0.5841531349       -0.1103818946
N                -5.2024854510        1.1928720123        0.6429526956
C                -4.7275482480        1.7373337641        1.8707889634
C                -5.5130132120        1.5276006594        3.1292654320
C                -5.4514470756        2.8895149347        2.4969771537
C                -0.2594774171       -2.0043947053        0.4647160156
C                 0.3547376654       -0.7039410285        0.9620145672
C                 6.0581592870        0.3801936693        0.2251799514
C                 7.1004765961        1.2049194351       -0.1422003615
F                 7.2892101640        1.4589232658       -1.4571594745
H                 9.4775883011        3.2501958952        1.1583950922
H                 8.7582462628        3.3328490489       -0.4614820987
H                 9.8964416372        2.0517319668       -0.0716521404
H                 8.3772298666        1.9404222329        2.8669233591
H                 6.5297667129        0.4862348365        3.5626506778
H                 4.6235917456       -0.8346613444        3.0650777873
H                 4.8332995959       -1.7811579866        1.5682258646
H                 3.1303002828        0.6159500239        1.7086065225
H                 1.6236294049        0.8092182393        0.0779186233
H                 1.5716953792        0.0220679988       -2.1387473491
H                 1.5722574465       -1.6652169565       -1.6245998291
H                -0.8111770367        0.2358436069       -1.3901405710
H                -0.6102447554       -0.9806653355       -2.6562057732
H                -1.6973802793       -3.6947546398       -0.7002531317
H                -0.7078531984       -3.3534437754       -2.1246761408
H                -2.8948122392       -1.0732753385       -1.0097220326
H                -4.8692146332       -2.3204503848       -3.8271518252
H                -6.9187534089       -1.2082069965       -4.6190946707
H                -7.7632193893        0.8034470526       -3.4526913024
H                -6.5819739714        1.6596794925       -1.4916822794
H                -6.2004498872        1.1662565545        0.5061709767
H                -3.6467257606        1.7037903565        1.9070682724
H                -4.9750403211        1.3577050361        4.0483293738
H                -6.4420462892        0.9815463169        3.0624874901
H                -6.3393747106        3.2582795214        2.0057133709
H                -4.8712138652        3.6600229592        2.9793077827
H                 0.5387993326       -2.7487848242        0.3152789904
H                -0.9593874522       -2.4012251935        1.2082727412
H                 0.7393561680       -0.8335776505        1.9755250476
H                -0.4267278321        0.0564260502        0.9828980462
H                 5.4240124622       -0.0544262936       -0.5320467134



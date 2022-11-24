%chk=./calculation/no_transfer_pubchem_sampling_6_b3lyp/69620375.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -4.9100677934        2.1266185538       -0.1799991846
C                -4.7219929040        1.1066366552        0.9456114031
N                -3.9026686674       -0.0184532948        0.5565695129
C                -4.5744586231       -1.0948765830       -0.1331729157
C                -2.5336721646        0.0697890840        0.4795220270
C                -1.8360693144        1.2038342063        0.9797025678
C                -0.4769098454        1.2880544475        0.9457027668
C                 0.3041200718        0.2453839838        0.4044454236
N                 1.6377707991        0.3665951602        0.3836350013
C                 2.3686188068       -0.5991540547       -0.1295001007
C                 3.8231366068       -0.3925971719       -0.1195156793
C                 4.3745041684        0.4758383024        0.8237710883
C                 5.7352837121        0.6998818405        0.8741172895
C                 6.5488102169        0.0491762975       -0.0365312575
F                 7.8837926083        0.2584834014        0.0054720646
C                 6.0341590252       -0.8056582241       -0.9934674715
C                 4.6695097997       -1.0226978699       -1.0301792954
C                 1.7891016404       -1.7845813669       -0.6448583424
C                 0.4315981491       -1.9288804768       -0.6284302849
C                -0.3757821725       -0.8959734009       -0.1020518003
C                -1.7738753641       -0.9600890840       -0.0577143699
H                -3.9479593489        2.5188875927       -0.5004434236
H                -5.3928715245        1.6603508419       -1.0357431800
H                -5.5295540587        2.9518369610        0.1613057040
H                -4.2840553041        1.5895071762        1.8216765268
H                -5.6973066130        0.7183319958        1.2520915154
H                -4.3001855250       -2.0580971344        0.3041151578
H                -5.6499908328       -0.9724222730       -0.0298041928
H                -4.3306087703       -1.1132612694       -1.2010428371
H                -2.3848734571        2.0319764430        1.3974888775
H                 0.0333262403        2.1589262834        1.3272251755
H                 3.7092861830        0.9695563120        1.5140164030
H                 6.1713030025        1.3653515342        1.6026110776
H                 6.6979009358       -1.2852291799       -1.6959045023
H                 4.2647654349       -1.6717065178       -1.7916418357
H                 2.4199966613       -2.5751125943       -1.0206539769
H                -0.0377101882       -2.8273324089       -1.0075861670
H                -2.2363715910       -1.8488951680       -0.4557677651



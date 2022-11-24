%chk=calculation/no_transfer_pubchem_sampling_5_b3lyp/98190274_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_5_b3lyp/98190274_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -5.2092731280        5.4362375653        2.5254797499
C                -4.0491355682        5.0708611655        1.9827880092
C                -3.3546678522        3.7757766033        2.2791159548
O                -3.0337884548        3.1538235911        1.0342029611
C                -2.3317224556        1.9822863385        1.0541744096
C                -2.0171527542        1.4415898067       -0.2033963427
C                -1.3096737809        0.2530741918       -0.2948856099
C                -0.8947680457       -0.4320035291        0.8578587012
C                -0.1222294987       -1.7442027858        0.7620939006
C                 1.1937813775       -1.6663979406       -0.0087069208
C                 2.2317934207       -0.8824670404        0.3948469964
O                 2.0204452467       -0.1263676512        1.5079431620
C                 3.5613711311       -0.7205380073       -0.2309460846
C                 4.3188735904       -1.8076160324       -0.6971007665
C                 5.5902055696       -1.6016020418       -1.2168902988
C                 6.1314276361       -0.3161045237       -1.2885351542
C                 5.3907624675        0.7775775033       -0.8059123860
C                 4.1193193096        0.5700844456       -0.2802328964
O                 5.8784228911        2.0601900276       -0.8470144167
C                 7.2885836110        2.1337992548       -1.0667058978
C                 7.6955064299        1.1878248068       -2.1842900743
O                 7.3788015802       -0.1607893072       -1.8279648003
C                 1.1188596243       -2.5420058956       -1.1791322037
O                 1.9171573307       -2.7517447814       -2.0771155727
C                -0.2363072986       -3.2677693368       -1.0817781270
O                -0.6509624681       -4.1337886626       -1.8370093691
N                -0.8638731574       -2.7841172325        0.0305554333
C                -2.1234798267       -3.3521296884        0.5120152133
C                -3.3852951910       -2.7076788468       -0.0392235262
C                -4.2439043771       -1.9840271173        0.7940924737
C                -5.4149126302       -1.4152203247        0.2892196396
C                -5.7383855982       -1.5656642662       -1.0592736105
C                -4.8869976480       -2.2894805544       -1.8982210822
C                -3.7197290260       -2.8608486586       -1.3929013568
C                -1.2132173856        0.1144690855        2.0988309263
C                -1.9270065937        1.3124313389        2.2127310270
H                -5.7525576557        4.7861999587        3.2083493621
H                -5.6608803236        6.4008942620        2.3118981617
H                -3.5268921190        5.7330891799        1.2936605957
H                -3.9938920109        3.1190678997        2.8859988158
H                -2.4260194398        3.9585120799        2.8433898906
H                -2.3455869030        1.9716916635       -1.0918108303
H                -1.0851973236       -0.1557864676       -1.2764397768
H                 0.0631813805       -2.0968770490        1.7878552262
H                 2.8651011879        0.2747587057        1.7727406090
H                 3.9085066459       -2.8075366058       -0.6655707222
H                 6.1839277475       -2.4304398948       -1.5888464160
H                 3.5597368385        1.4429551995        0.0452275615
H                 7.8225459907        1.8750745672       -0.1419387081
H                 7.5050384211        3.1731940807       -1.3258277034
H                 8.7742429911        1.2136707913       -2.3568470837
H                 7.1750708250        1.4547271318       -3.1136256990
H                -2.1185446952       -3.2916960572        1.6058627885
H                -2.1006827958       -4.4103015548        0.2296552223
H                -3.9913292997       -1.8558753222        1.8441436272
H                -6.0708652238       -0.8532099664        0.9490726233
H                -6.6489493243       -1.1232142797       -1.4549987937
H                -5.1359605363       -2.4149332447       -2.9489150241
H                -3.0580471970       -3.4299950188       -2.0398467843
H                -0.8934510587       -0.3941999930        3.0058149143
H                -2.1573295988        1.7027654344        3.1974450826



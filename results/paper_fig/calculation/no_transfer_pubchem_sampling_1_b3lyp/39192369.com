%chk=./calculation/no_transfer_pubchem_sampling_1_b3lyp/39192369.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
O                -1.7251987944        0.7820139629        3.5097756614
C                -0.5809117869        0.3923600878        3.3495485051
C                -0.1951601680       -1.0799581750        3.5099259908
C                 0.1421005422       -1.5937947891        2.1496220174
C                -0.4757547581       -2.5738905232        1.4425057869
S                 0.2040518635       -2.7082151082       -0.1436261595
C                 1.2863629785       -1.3812097362        0.2829007820
N                 2.1512809162       -0.8566526255       -0.6288122919
C                 2.7679254734        0.3723946660       -0.4563933361
O                 2.8045852885        0.9491966049        0.6039108254
C                 3.4165244905        0.8763268940       -1.6935594774
C                 4.6547897343        1.5036903759       -1.5911405283
C                 5.2789446607        1.9952031847       -2.7224537790
C                 4.6608501838        1.8848515160       -3.9576381653
C                 3.4129239193        1.2913592880       -4.0588434296
C                 2.7901940306        0.7828861506       -2.9331360730
N                 1.1398552137       -0.9532352765        1.4800239791
N                 0.4449647317        1.1931379101        3.0126280869
C                 0.2128759521        2.5616042020        2.6138657409
C                -0.0335142113        2.6713919104        1.1032312587
N                -1.1244060067        1.8063488490        0.7162328167
C                -1.0065542799        0.8735693478       -0.2435887868
O                -0.0560608288        0.8150229117       -1.0098244613
C                -2.1303806447       -0.1121776706       -0.3465754463
C                -2.3081390908       -0.7430165973       -1.5750044418
C                -3.2627448704       -1.7277468949       -1.7365852315
C                -4.0441748529       -2.0931002574       -0.6496846305
C                -5.0762669849       -3.1779077954       -0.7654602595
F                -5.1759947152       -3.6932639043       -1.9975488824
F                -6.3074058429       -2.7480629876       -0.4324518536
F                -4.8121254033       -4.2119882148        0.0576027570
C                -3.8754728195       -1.4753816880        0.5835248388
C                -2.9219990249       -0.4858493138        0.7365880856
H                 0.6763475269       -1.1537802691        4.1618651226
H                -1.0307923786       -1.6307926783        3.9385869270
H                -1.2751138283       -3.2203499590        1.7397818721
H                 2.1215447805       -1.2449677931       -1.5604528045
H                 5.1171866609        1.5901435781       -0.6194713485
H                 6.2452767546        2.4699833694       -2.6414830098
H                 5.1463344621        2.2726369148       -4.8408887715
H                 2.9223129948        1.2273661550       -5.0187449717
H                 1.7971804437        0.3647055651       -3.0057457283
H                 1.2767707830        0.7523207982        2.6288412641
H                -0.6639630506        2.9176046589        3.1625370930
H                 1.0826320184        3.1626353256        2.8834479587
H                 0.8525408847        2.3471272415        0.5519911700
H                -0.2610475141        3.7101391053        0.8414471941
H                -1.9170462271        1.7869275353        1.3412459562
H                -1.6743754655       -0.4513440224       -2.3986116626
H                -3.3956353414       -2.2150013718       -2.6899236686
H                -4.4860393451       -1.7761197384        1.4214595280
H                -2.7900750539       -0.0451417189        1.7130689812



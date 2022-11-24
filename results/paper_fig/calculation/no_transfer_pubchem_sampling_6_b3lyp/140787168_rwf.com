%chk=calculation/no_transfer_pubchem_sampling_6_b3lyp/140787168_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_6_b3lyp/140787168_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C               -10.5510780443        1.1117159370        3.2962854743
C                -9.7545526396        1.1264047695        2.0121674181
C               -10.2912928858        1.6700824438        0.8359494501
C                -9.5710129594        1.6756072127       -0.3546489865
C                -8.2807259587        1.1266975228       -0.4050453215
S                -7.4493035996        1.1615466949       -1.9973304248
C                -5.7832169103        0.6139579665       -1.6394414312
C                -5.4145919873       -0.7186153847       -1.8540212587
C                -4.1033092592       -1.1420362672       -1.6347262695
C                -3.1265416648       -0.2367085218       -1.1924088908
S                -1.4318113019       -0.6595007488       -0.8797629199
C                -1.3223952419       -2.4378749911       -1.2549949223
C                 1.9200256648       -2.9053342595       -0.6239761218
C                 2.4298111270       -3.2230597252        0.6468497023
C                 3.8003890897       -3.2135317510        0.9611272996
C                 4.6889358803       -2.8348137914       -0.0351109755
O                 6.0804527418       -2.8243862849        0.2739151222
C                 6.7097375019       -1.6499114844        0.3734212490
C                 8.1091204594       -1.7040899711        0.5812499527
C                 8.8436752224       -0.5436571493        0.7148228751
C                 8.1914659058        0.7075405952        0.6507585624
S                 9.1927110792        2.1319020950        0.8147313522
C                 8.8084787486        3.2241048119       -0.6271374463
C                 8.4222544038        3.1703770861        2.1356330387
C                 6.7947714823        0.7705393485        0.4575720048
C                 6.0621154220       -0.3955167643        0.3185335075
C                 4.2589046104       -2.4944332594       -1.3160731981
C                 2.8844618278       -2.5391332393       -1.5857651684
C                -3.5019394029        1.1046557188       -0.9805228797
C                -4.8067860925        1.5243944607       -1.2044314004
C                -7.7318564450        0.5807061846        0.7580144462
C                -8.4654094457        0.5887507291        1.9464194757
H               -11.0089239651        2.0876219106        3.4984068115
H               -11.3667497973        0.3772412583        3.2566478190
H                -9.9206088507        0.8543463046        4.1537190131
H               -11.2896550764        2.1032790644        0.8522382808
H               -10.0092512148        2.1123315271       -1.2488345647
H                -6.1598628473       -1.4305648870       -2.1968955931
H                -3.8468573488       -2.1808666993       -1.8101518375
H                -1.5696902138       -2.6334263267       -2.3030238544
H                -1.9760469530       -3.0184859105       -0.5966643249
H                -0.2576532790       -2.7020270925       -1.0617641725
H                 1.7475512326       -3.4953329460        1.4572006622
H                 4.1765729905       -3.4814386784        1.9477223298
H                 8.5858176014       -2.6773535724        0.6270732437
H                 9.9183357977       -0.5937273059        0.8667252133
H                 9.3574051409        4.1598821134       -0.4947647539
H                 9.1609061185        2.7026501854       -1.5183448688
H                 7.7344961052        3.4085651876       -0.6919994307
H                 8.5160090755        2.6121465214        3.0683500615
H                 7.3717648582        3.3623792756        1.9090362082
H                 8.9843960714        4.1056473898        2.1968550943
H                 6.2756026406        1.7243843821        0.4172707587
H                 4.9899377669       -0.3594922745        0.1694975411
H                 4.9909958645       -2.2181164703       -2.0755879491
H                 2.5726484754       -2.2859213780       -2.6029981655
H                -2.7574599860        1.8210999459       -0.6421612499
H                -5.0777409048        2.5637437119       -1.0433363388
H                -6.7349512517        0.1530302223        0.7424159078
H                -8.0184723788        0.1620225573        2.8421328436



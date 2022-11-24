%chk=calculation/no_transfer_pubchem_sampling_1_b3lyp/59929573_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_1_b3lyp/59929573_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -1.4385672305       -5.5868597286        1.0476309324
O                -0.9712201891       -4.2505394103        0.9386761684
C                 0.1682512064       -3.8929646827        1.5957481890
C                 0.9138498746       -4.7331770729        2.4241995128
C                 2.0659076147       -4.2567272619        3.0505616452
C                 2.4943057564       -2.9469408908        2.8352708205
C                 3.7418629012       -2.4743913745        3.5265238247
O                 4.1411564133       -2.9851729779        4.5704312591
N                 4.3938284497       -1.4421277945        2.9049525484
C                 5.5611958766       -0.7736625837        3.4561024119
C                 5.2605531339        0.6621622616        3.9246378479
C                 4.5456482423        1.5085316777        2.8768712090
O                 5.3345102525        1.5012063716        1.6864905506
C                 4.8619150374        2.1465777018        0.5700778760
C                 3.5877645079        2.7047117716        0.4636089211
C                 3.2026355710        3.3319868481       -0.7269677653
C                 4.0674228634        3.4157589473       -1.8177994851
C                 3.6641102910        4.1092113944       -3.0989486950
C                 5.3421320624        2.8417432672       -1.6824984933
C                 5.7628094501        2.2092839513       -0.5149202826
C                 7.1371945863        1.6010445595       -0.3912916430
C                 1.7384809438       -2.0977022625        2.0138373138
C                 0.5761083038       -2.5522788784        1.3927768829
N                -0.2382638063       -1.7559710038        0.5566347482
C                -0.1106946916       -0.4174686467        0.3299732244
O                 0.7708945327        0.3025821780        0.7896537478
C                -1.1228785706        0.2145130453       -0.6557131692
C                -0.7203947215        0.0119578032       -2.1461454247
C                 0.7496295404        0.1452843561       -2.4726471849
O                -1.5728309310       -0.2034908138       -2.9786413498
N                -2.5291714488       -0.0883721499       -0.4522980018
C                -3.1733442662       -1.2732567114       -0.7039290698
O                -2.6612085159       -2.3614750936       -0.9100193422
C                -4.6894544828       -0.9795393138       -0.6604986559
O                -5.3086119579       -1.1975039922       -1.9028139349
C                -5.5902174939       -2.5658758958       -2.1883700302
N                -4.7185714996        0.4179931837       -0.3276768361
C                -5.9398945097        1.2038730458       -0.1289922718
C                -6.5521157149        1.7768662758       -1.3942854839
C                -7.7554165913        1.2720940935       -1.8957646777
C                -8.3294359663        1.8139306152       -3.0470037764
C                -7.6992707538        2.8670241704       -3.7105804545
C                -6.4963467810        3.3769207478       -3.2159083562
C                -5.9275220297        2.8387915581       -2.0622264712
C                -3.4694739629        0.9703961689       -0.2706181149
O                -3.1678214236        2.1307586919       -0.0742532998
H                -2.3332529609       -5.6363841791        0.4255435582
H                -1.6968100172       -5.8360499371        2.0846814170
H                -0.6929803147       -6.3013312891        0.6767877279
H                 0.5977478707       -5.7571103182        2.5879759099
H                 2.6431764517       -4.8957210784        3.7102572812
H                 4.0930748719       -1.1702907793        1.9807099883
H                 6.3497370204       -0.7539105918        2.6967789863
H                 5.9072747341       -1.3813313694        4.2948301111
H                 4.6250819954        0.6323487574        4.8185218404
H                 6.2047403293        1.1437062165        4.2082274572
H                 3.5501711013        1.0930659742        2.6722529343
H                 4.4104619831        2.5406417895        3.2295603054
H                 2.8812929360        2.6477150795        1.2829908859
H                 2.2077188313        3.7670843390       -0.7915124079
H                 2.6117315800        4.4105276117       -3.0742116098
H                 3.8047897353        3.4605388183       -3.9726650898
H                 4.2611442882        5.0138163630       -3.2740021477
H                 6.0367921627        2.8924848010       -2.5197293085
H                 7.7184926118        1.7624346006       -1.3042932438
H                 7.0839375332        0.5208527062       -0.2068256329
H                 7.6904732615        2.0323952196        0.4512779320
H                 2.0101294964       -1.0623531490        1.8724430354
H                -0.9986708497       -2.2476149031        0.0819377096
H                -1.0356881079        1.2882357742       -0.4576121370
H                 0.8663135339        0.2753538337       -3.5501371247
H                 1.2188624813        0.9704240676       -1.9274667224
H                 1.2722094647       -0.7711256008       -2.1692936863
H                -5.1713454689       -1.5981692521        0.1166413399
H                -6.1965339876       -3.0187758007       -1.3885426785
H                -4.6731603975       -3.1493927233       -2.3177323181
H                -6.1621806585       -2.5652470629       -3.1181340171
H                -6.6610191871        0.5559762229        0.3808826924
H                -5.6702959259        2.0101499928        0.5594823071
H                -8.2466179590        0.4473527787       -1.3840057901
H                -9.2661781870        1.4125999194       -3.4247878906
H                -8.1434914441        3.2904773493       -4.6074809762
H                -6.0027103464        4.1994365097       -3.7266685826
H                -4.9938563341        3.2374861635       -1.6745804193



%chk=./calculation/no_transfer_pubchem_sampling_0_b3lyp/123763878.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
O                -2.3635416476       -0.8012904985        2.0182616763
C                -2.1889520210        0.0850561575        1.2030045697
C                -0.8635257228        0.5698871941        0.8045843435
C                 0.4130516189        0.0712933678        1.0038582988
C                 1.2562104913        1.0155563566        0.4014578926
C                 2.7442837699        0.9898265618        0.3057811961
O                 3.1261543187       -0.2585175382       -0.2417398117
C                 4.4446669920       -0.5191387676       -0.4150325603
C                 5.4837254297        0.3554185697       -0.1088596559
C                 6.7976600331       -0.0245788464       -0.3294162139
C                 7.1068575405       -1.2654196248       -0.8528482528
C                 6.0603273047       -2.1210101938       -1.1519234683
F                 6.3369414899       -3.3385542493       -1.6651158236
C                 4.7433223612       -1.7759500484       -0.9447792035
N                 0.5496561031        2.0138549451       -0.1187263684
N                -0.7152936078        1.7327427282        0.1265944469
C                -1.8237890533        2.5686323512       -0.2336137517
C                -3.0060794609        1.6450169017       -0.5131692174
N                -3.2231826339        0.7547688512        0.6111436035
C                -4.5770666910        0.3144348607        0.8779593142
C                -5.1211988071       -0.5772739447       -0.2151847530
F                -5.5198403594        0.1584752650       -1.3139591821
C                -4.5293427783       -1.9201083476       -0.4955968179
C                -5.9184664078       -1.8078737350        0.0718300624
H                 0.6882861026       -0.8386201318        1.4927916015
H                 3.0703379776        1.8187597845       -0.3346204214
H                 3.1843066387        1.1115527382        1.3071722435
H                 5.2829899057        1.3329130949        0.2989187391
H                 7.5924947160        0.6643018623       -0.0869526756
H                 8.1247656628       -1.5718367517       -1.0287163094
H                 3.9456889740       -2.4590891402       -1.1849578349
H                -1.5571469384        3.1565236963       -1.1131810364
H                -2.0563028263        3.2518787626        0.5930908452
H                -2.8121829545        1.0657255204       -1.4263413489
H                -3.9140274116        2.2298294340       -0.6695645889
H                -5.2233992612        1.1908878151        0.9914108097
H                -4.5458760417       -0.2458128393        1.8146326044
H                -3.7318895434       -2.2543467308        0.1494044011
H                -4.4284306952       -2.2190833133       -1.5274201029
H                -6.7478319133       -2.0210845085       -0.5845985898
H                -6.0843636541       -2.0777476091        1.1024363401



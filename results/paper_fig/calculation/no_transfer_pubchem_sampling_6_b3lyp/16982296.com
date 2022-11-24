%chk=./calculation/no_transfer_pubchem_sampling_6_b3lyp/16982296.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -7.8578821963        2.3593848972        1.0312053024
C                -7.5933112149        1.9367801550       -0.3828496064
C                -8.4844314446        2.2514023353       -1.3968379399
C                -8.2164023504        1.8725790185       -2.7024053912
C                -7.0635503892        1.1811512693       -3.0070657240
C                -6.1571889936        0.8588067179       -1.9965391660
O                -5.0472912941        0.1727101700       -2.3955602034
C                -4.0816261725       -0.1840580157       -1.4314302185
C                -2.9709342519       -0.9336269152       -2.1633382420
C                -1.9247327381       -1.5240726772       -1.2226950265
C                -1.0679026268       -0.4548743025       -0.5344975585
N                -0.0734790163       -1.0516657652        0.3134222432
C                 1.1869203162       -1.4887331731       -0.0124921619
C                 1.7816849058       -1.3574845482       -1.3729819961
C                 3.2937085649       -1.1476898871       -1.2823896478
C                 3.6177905898        0.2564675844       -0.7787387908
N                 5.0345975478        0.4322521278       -0.5679668718
C                 5.5390922326        0.9600677145        0.5735231201
O                 4.8361349613        1.3796801245        1.4715144080
C                 7.0311713807        0.9871764612        0.6437082626
C                 7.6258903673        0.8322975105        1.8919709045
C                 9.0019877946        0.8568301668        2.0196029378
C                 9.7971520027        1.0533550103        0.9024643448
C                 9.2113896512        1.2316017475       -0.3399181022
C                 7.8345213481        1.1965536878       -0.4726614531
N                 1.8074015341       -2.0472935116        0.9857511861
C                 0.9411020718       -2.0032202651        2.0460492467
C                 1.0875649529       -2.4555305009        3.3517178442
C                 0.0238276236       -2.2810886349        4.2115771053
C                -1.1622114007       -1.6732041729        3.7967166197
C                -1.3269358173       -1.2147792527        2.5047750140
C                -0.2616916264       -1.3843861241        1.6357880529
C                -6.4295167285        1.2397313534       -0.6895549868
H                -8.9229364401        2.4944695010        1.2021217830
H                -7.3600228374        3.3080281499        1.2353424085
H                -7.4778577090        1.6230592326        1.7360111514
H                -9.3902445841        2.7921770485       -1.1662269911
H                -8.9149955646        2.1184936104       -3.4883361571
H                -6.8386788000        0.8761487839       -4.0165491178
H                -4.5289396036       -0.8341922300       -0.6654486609
H                -3.6884167584        0.7182625606       -0.9434020039
H                -2.4977954476       -0.2546597144       -2.8748094601
H                -3.4377334824       -1.7376219340       -2.7345218238
H                -1.2643890209       -2.1802402553       -1.7937827046
H                -2.4143465911       -2.1314697663       -0.4581138365
H                -0.5791671244        0.1709849468       -1.2864782230
H                -1.6922748492        0.1892740588        0.0891545095
H                 1.3228747806       -0.5260684855       -1.9123898399
H                 1.5758504501       -2.2743452723       -1.9339819181
H                 3.7442770308       -1.2977265532       -2.2667823189
H                 3.7011092742       -1.8821752135       -0.5878699236
H                 3.2543059561        1.0056805898       -1.4960304851
H                 3.1366010621        0.4346362719        0.1863713179
H                 5.6584973207        0.0396943351       -1.2540817030
H                 6.9881434937        0.6932449660        2.7517353084
H                 9.4562404447        0.7278017078        2.9908247371
H                10.8725987747        1.0770346648        0.9997865680
H                 9.8301894166        1.4019163761       -1.2090343596
H                 7.3903138631        1.3643580895       -1.4440180290
H                 2.0056045422       -2.9229864448        3.6686872188
H                 0.1039464580       -2.6213286873        5.2334966030
H                -1.9701714308       -1.5596727648        4.5045056937
H                -2.2495245156       -0.7466045582        2.1927507255
H                -5.7459066923        1.0012306808        0.1112060258



%chk=./calculation/no_transfer_pubchem_sampling_3_b3lyp/112850755.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 5.7691301298       -0.1177546473        3.4604457163
O                 5.8947517213       -0.5176470537        2.1186789248
C                 4.8876586419       -0.2090345597        1.2572471796
C                 3.7375399941        0.4863499864        1.5835165894
C                 2.7640984298        0.7520954720        0.6298943792
C                 2.9264202675        0.3028979196       -0.6688690564
N                 1.9875200878        0.5495057963       -1.6757986601
C                 0.6389127638        0.6075049924       -1.5928423841
C                -0.1124538535        0.2265459752       -0.4648743937
C                -1.4795097898        0.3500991820       -0.5819961916
C                -2.3844258847       -0.0293090447        0.5503323302
O                -1.9751160403       -0.4246918163        1.6221176060
N                -3.6867591628        0.1388775301        0.1926465438
C                -4.8405335975       -0.0904731537        0.9179400205
C                -4.8307582215       -0.5461733103        2.2344056469
C                -6.0200652736       -0.7551200180        2.9062286584
C                -7.2406663348       -0.5173104966        2.2889299674
C                -7.2718037464       -0.0668229570        0.9847844851
C                -6.0804798811        0.1487323156        0.2924403528
C                -6.1222333485        0.6051072691       -1.0421126703
N                -6.1532775923        0.9755461075       -2.1281726637
N                -2.0759170292        0.7998523712       -1.6931487906
C                -1.2914049416        1.1039131844       -2.7152541886
C                -1.9501812535        1.5929642356       -3.9609423190
N                 0.0287590197        1.0188131163       -2.7105960344
C                 4.0952053655       -0.3807524319       -1.0176145933
C                 5.0733329828       -0.6409937490       -0.0774997930
O                 6.2374218246       -1.2970708381       -0.3232173970
C                 6.4855660993       -1.7585177091       -1.6273794022
H                 6.6684648202       -0.4768882570        3.9581309845
H                 4.8899190922       -0.5663520972        3.9346288354
H                 5.7158033307        0.9721599336        3.5536806427
H                 3.5815346419        0.8434082260        2.5882680987
H                 1.8984210263        1.3304929439        0.9072728438
H                 2.3434984116        0.6900792811       -2.6119485048
H                 0.3270299055       -0.1770389173        0.4297144965
H                -3.7850910946        0.4890838105       -0.7626780064
H                -3.8845415542       -0.7318215295        2.7150943228
H                -5.9950784588       -1.1083126859        3.9261081199
H                -8.1628177635       -0.6837619284        2.8240134107
H                -8.2106115160        0.1232722268        0.4856157221
H                -1.3881957349        1.2677225573       -4.8312510292
H                -2.9739887556        1.2336298741       -4.0042459177
H                -1.9662253887        2.6818870001       -3.9514126423
H                 4.2144695155       -0.7055630377       -2.0393387179
H                 7.4571552303       -2.2479065020       -1.5846955265
H                 5.7305314227       -2.4832820277       -1.9498770436
H                 6.5289904927       -0.9334015390       -2.3463699510



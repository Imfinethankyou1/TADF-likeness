%chk=./calculation/no_transfer_pubchem_sampling_4_b3lyp/44030558.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -6.0326415267        0.5641723087       -2.6958463632
O                -7.0847363390        1.1431555446       -1.9488210810
C                -6.8886540045        1.4611157614       -0.6545548500
C                -7.9868966189        1.9821589452        0.0296283820
C                -7.9123818363        2.3569239932        1.3598573695
C                -6.7340793501        2.2347417455        2.0755227788
C                -5.6365697651        1.7261008784        1.4102379428
S                -4.0336123312        1.4776758999        2.0457490190
C                -3.5703104806        0.8447916980        0.4305107738
N                -2.2564282162        0.5143419257        0.1863302994
C                -1.5285160930       -0.2137550466        1.2048452000
C                -1.7462137023       -1.7375053496        1.0401009387
C                -3.1672932310       -2.2037565994        1.4077117796
C                -3.5345840086       -3.1490609233        0.2643867874
C                -2.7375243115       -2.5451352582       -0.8883709755
O                -1.5159482117       -2.1171549158       -0.3065793353
C                -1.7650800619        0.6501819921       -1.1045783784
O                -2.4639135743        0.9303409151       -2.0476200917
C                -0.2816287487        0.5182910796       -1.2485598657
C                 0.5685404021        1.4149912933       -0.6034889767
C                 1.9335025992        1.3439390976       -0.8165429729
C                 2.4356732303        0.3806579771       -1.6703952508
S                 4.2112951789        0.3108660707       -1.9169508654
O                 4.6732881733        1.5907250835       -2.3534845319
O                 4.5167228315       -0.8322453110       -2.7307158608
N                 4.6953394774        0.0485363305       -0.3182591787
C                 4.4429955754       -1.3162997269        0.0976948643
C                 5.6669933706       -2.1918085212       -0.1464202238
C                 6.8092092676       -1.6756874518        0.7212702615
C                 6.9335077659       -0.1772311009        0.6420167323
C                 8.0865313660        0.4435298574        1.1048149134
C                 8.2331227459        1.8166834920        1.0565513764
C                 7.2080812648        2.5947459440        0.5445182601
C                 6.0460722121        1.9996381892        0.0935848816
C                 5.9029033344        0.6155745881        0.1309930291
C                 1.6025301827       -0.4983439494       -2.3335455496
C                 0.2348507884       -0.4225885741       -2.1308141269
N                -4.5069701151        0.8327459866       -0.4270980079
C                -5.6855641696        1.3258418766        0.0547420336
H                -5.7764345079       -0.4247394370       -2.3073569922
H                -6.4174104211        0.4714219264       -3.7103056346
H                -5.1344470585        1.1832339984       -2.6896785977
H                -8.9086577344        2.0879065634       -0.5195617897
H                -8.7893611520        2.7547557811        1.8465213179
H                -6.6667320672        2.5292641801        3.1099268565
H                -0.4642679441       -0.0020251701        1.1066317351
H                -1.8688336493        0.1051110676        2.1923364046
H                -0.9913517738       -2.2456353128        1.6551317620
H                -3.1931182679       -2.6860617177        2.3820526854
H                -3.8519871631       -1.3566560004        1.4236021331
H                -3.1801059471       -4.1585377975        0.4748518097
H                -4.6024129532       -3.1780278835        0.0642832699
H                -3.2728155142       -1.6926794106       -1.3290145643
H                -2.4905194495       -3.2590739554       -1.6749722477
H                 0.1598853893        2.1720318208        0.0505827624
H                 2.6156221874        2.0245795142       -0.3327539337
H                 3.5815713328       -1.6719783454       -0.4701560808
H                 4.1845038204       -1.3041213252        1.1626065754
H                 5.9249921378       -2.1213192464       -1.2041983606
H                 5.4427718724       -3.2326098457        0.0899162064
H                 7.7531691703       -2.1349274134        0.4205857314
H                 6.6295730861       -1.9562952297        1.7653027419
H                 8.8833882449       -0.1704299515        1.5007865901
H                 9.1415268777        2.2770031592        1.4147641907
H                 7.3101727480        3.6684982547        0.5020602274
H                 5.2482319501        2.5974003803       -0.3137927389
H                 2.0388068442       -1.2235531503       -3.0024010076
H                -0.4373741288       -1.0944301995       -2.6391791889



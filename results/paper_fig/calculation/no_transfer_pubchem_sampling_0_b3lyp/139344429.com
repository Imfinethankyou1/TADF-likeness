%chk=./calculation/no_transfer_pubchem_sampling_0_b3lyp/139344429.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
O                -4.3289896854       -1.2934018868        1.2400028618
C                -4.4066005225       -0.3941162457        0.4307649894
C                -5.5556910548        0.5959659256        0.4630070750
C                -6.2273033523        0.8123952481        1.8063997452
C                -5.3810480537        1.8724732371        1.2147057702
F                -5.9314910192        3.0077646331        0.7295095258
F                -4.1808416249        2.1960846501        1.7357945980
N                -3.5026886501       -0.2217954682       -0.5579056967
C                -3.3893182434        0.9343510019       -1.4128534123
C                -2.1296985209        1.7214072351       -1.0405567271
N                -0.9872920036        0.8322304855       -1.0204892875
C                 0.2086234292        1.2189848887       -0.5245000379
C                 0.4682367083        2.5405010414       -0.1012480999
C                 1.7366562555        2.7938004615        0.3650250079
N                 2.6903925629        1.8710493423        0.4241188922
C                 2.3587758685        0.6528812805        0.0005618234
C                 3.3586362350       -0.3877549617        0.0403976775
C                 3.2579370839       -1.7181944324       -0.3477405138
N                 4.4068001278       -2.3758890943       -0.1610799766
C                 5.2549632230       -1.4951199151        0.3432176473
C                 6.5937274980       -1.6493521627        0.7248639629
C                 7.2713662952       -0.5743802953        1.2257180222
C                 6.6112399433        0.6720991548        1.3503125540
C                 7.3862167410        1.8320191155        1.9010297410
F                 8.4608248633        2.1259908181        1.1450698992
F                 6.6675283485        2.9572957014        1.9917944807
F                 7.8630361089        1.5824723192        3.1344422062
C                 5.3046436793        0.8234410061        0.9786917366
N                 4.6451632548       -0.2426670922        0.4857365293
N                 1.1692255848        0.2927241120       -0.4680399534
C                -1.2213916337       -0.5378691334       -1.4036438998
C                -2.3686850269       -1.1476534119       -0.5865717281
C                -2.7688162274       -2.4708998539       -1.1574420067
C                -2.0756569016       -3.6907353026       -1.0339242230
N                -2.6950731033       -4.6539932646       -1.6804033624
N                -3.7709363477       -4.0958578351       -2.2200701773
C                -3.8617514496       -2.7798854525       -1.9371573049
H                -6.1718451229        0.6264145513       -0.4269935523
H                -7.2928266595        0.9729130853        1.8419799133
H                -5.8164427557        0.2543099370        2.6349713258
H                -3.3419190395        0.6087811574       -2.4571156631
H                -4.2561169774        1.5829375413       -1.3004583091
H                -2.2707679768        2.1647671812       -0.0471839391
H                -1.9760825114        2.5312695395       -1.7658302418
H                -0.2744554927        3.3170195687       -0.1388595839
H                 2.0232617394        3.7786436105        0.7112767773
H                 2.3957439758       -2.2089413012       -0.7525773847
H                 7.0514354152       -2.6171104842        0.6110214529
H                 8.3025549786       -0.6548543646        1.5299595249
H                 4.7467808168        1.7463816911        1.0492764989
H                -1.4493587250       -0.5962048070       -2.4746597055
H                -0.3014858980       -1.0912713669       -1.2153554112
H                -2.0393348252       -1.2979518867        0.4501749197
H                -1.1675908116       -3.8854168400       -0.5010403225
H                -4.4149561404       -4.6593585860       -2.7519532420
H                -4.6773163795       -2.1786930764       -2.2846893961



%chk=./calculation/no_transfer_pubchem_sampling_4_b3lyp/20547649.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 6.2535751521        1.7910650187        1.4153239741
C                 5.1459061714        1.6242698671        0.3808666516
C                 4.4448505079        0.2764894606        0.5453008636
C                 3.3114765861        0.0000115326       -0.4520801516
C                 3.7797557252       -0.0998799428       -1.9130704282
C                 4.7549578071       -1.2449467925       -2.2273641997
C                 6.2118711104       -0.7983625526       -2.2972668312
C                 2.5820388806       -1.3045983116       -0.0412010786
O                 1.5042538145       -1.0372384840        0.8307564582
C                 0.4480101757       -0.3286794627        0.2083884891
C                -0.2842360815        0.3900425117        1.3636704443
C                -1.0577336939        1.6334605720        0.9145588171
N                -1.9381965591        2.0822573690        1.9574012039
C                -3.1947239313        1.6051066337        2.2030441910
C                -3.6337958856        2.2579860095        3.3182404477
N                -2.6780716272        3.1159215823        3.7616867716
C                -1.6798094852        2.9832286113        2.9326577283
C                -0.4521857212       -1.3175919390       -0.5918490582
C                -1.9435000816       -1.1194976515       -0.4908174968
C                -2.5901720834       -0.3202128380       -1.4400781952
C                -1.8031375545        0.4454513230       -2.4707131968
C                -3.9833803089       -0.2335380020       -1.4514817810
C                -4.6492594464        0.5840850031       -2.5244159241
C                -4.7335697037       -0.8893621054       -0.4728164051
C                -6.2361394275       -0.8039376414       -0.5160616682
C                -4.0836529519       -1.6140773358        0.5290890894
C                -4.8976646385       -2.2506053308        1.6255500569
C                -2.6910511635       -1.7575179770        0.5046179973
C                -2.0377826105       -2.6116262799        1.5620499267
O                 0.9675797314        0.6007215851       -0.7256818960
C                 2.2337123438        1.1032826403       -0.3534406492
H                 6.9993502943        1.0060381793        1.3083101377
H                 5.8472565955        1.7390039465        2.4229875492
H                 6.7485142511        2.7517475995        1.2944715950
H                 5.5777018220        1.7019842459       -0.6175765945
H                 4.4247095492        2.4352415043        0.4943345736
H                 4.0227163188        0.2218708268        1.5527433773
H                 5.1846492055       -0.5220440546        0.4622281292
H                 2.8789076129       -0.2164347209       -2.5209948730
H                 4.2361486754        0.8476709162       -2.2042293592
H                 4.4894790354       -1.6767689294       -3.1948070343
H                 4.6549948545       -2.0366866921       -1.4841722975
H                 6.5504553908       -0.3956687651       -1.3474097601
H                 6.3376732856       -0.0319483786       -3.0592918215
H                 6.8499358130       -1.6409544212       -2.5556104205
H                 3.2402338285       -1.9701364100        0.5193572583
H                 2.2186763931       -1.8316979322       -0.9323335871
H                 0.4614344209        0.6675190608        2.1116864300
H                -0.9701578385       -0.3112595689        1.8310120432
H                -0.3666585795        2.4410120781        0.6606783948
H                -1.6562080703        1.3960314841        0.0345659088
H                -3.6498912601        0.8686688038        1.5729959111
H                -4.5725771826        2.1697883337        3.8273563583
H                -0.7572423313        3.5261986664        2.9945522730
H                -0.1505808997       -1.2527174604       -1.6368819493
H                -0.1919064395       -2.3201917338       -0.2528164523
H                -2.2808314669        1.3977496973       -2.6866693575
H                -0.7937146893        0.6528756756       -2.1316237941
H                -1.7489962877       -0.1183229692       -3.4035352251
H                -4.2213638399        0.3542954352       -3.4981802903
H                -4.5032075943        1.6482941128       -2.3331382629
H                -5.7159682483        0.4028761279       -2.5751578872
H                -6.7032784958       -1.4271883336        0.2362444131
H                -6.5659728599        0.2234028695       -0.3622489129
H                -6.6024498358       -1.1391185243       -1.4856437196
H                -5.4324002712       -3.1221407292        1.2466578761
H                -5.6248803575       -1.5436185738        2.0202586481
H                -4.2809952776       -2.5745618436        2.4548344709
H                -0.9597837279       -2.6556358126        1.4587812345
H                -2.4267842363       -3.6285455428        1.5132401218
H                -2.2470345676       -2.2185696845        2.5572926290
H                 2.1930369276        1.4986555409        0.6686300216
H                 2.4370870325        1.9215819033       -1.0457579066



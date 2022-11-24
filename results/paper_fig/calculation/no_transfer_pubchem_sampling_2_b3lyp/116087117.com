%chk=./calculation/no_transfer_pubchem_sampling_2_b3lyp/116087117.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 4.2599884946       -1.0842396630       -2.5412518303
C                 3.4204309997       -0.9400640977       -1.3306987692
N                 2.6098539804        0.0266105438       -1.1068822552
C                 1.9638854198       -0.0906290702        0.0846156560
C                 0.9863189054        0.9619451801        0.5114252606
C                 1.3899712127        1.6064179041        1.8719194787
N                 0.4417638263        1.1136830630        2.8433631280
C                -0.4855967089        0.5022613278        2.2108560437
N                -1.5915328870       -0.0417834325        2.8047844377
N                -0.3202234822        0.4096082685        0.8449441894
C                -1.2702942610        0.0038981239       -0.1010270586
C                -0.9314598231       -0.9767833360       -1.0312036654
C                -1.8479670710       -1.4065969773       -1.9703747663
C                -3.1217333050       -0.8649570463       -1.9851148446
C                -3.4539288066        0.1264042353       -1.0801699965
C                -2.5434169557        0.5903980671       -0.1397115419
C                -2.9229733197        1.7154475482        0.7731768059
C                 2.2998107058       -1.1978631536        0.8019735781
S                 3.4659557291       -2.1386347531       -0.0424910901
H                 4.0330644806       -2.0183597559       -3.0532372574
H                 4.0666434540       -0.2492136084       -3.2088997843
H                 5.3153521363       -1.0954342457       -2.2721258143
H                 0.9023321414        1.6940551197       -0.2975902341
H                 1.3411029709        2.6998244670        1.8316070857
H                 2.4025518396        1.3238474507        2.1680650647
H                -1.9852716453       -0.8446813323        2.3387658426
H                -1.5022432145       -0.1264859718        3.8045298240
H                 0.0613081472       -1.3989205216       -1.0121018898
H                -1.5692790819       -2.1674628031       -2.6840331844
H                -3.8485727157       -1.2021334268       -2.7088154072
H                -4.4390033287        0.5701895174       -1.1125705355
H                -3.3299754084        1.3367683893        1.7101614412
H                -2.0560459924        2.3256734544        1.0171434430
H                -3.6755550757        2.3424496983        0.3002419397
H                 1.9347376391       -1.5052391637        1.7607267061



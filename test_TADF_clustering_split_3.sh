python train.py  --tdata 'data/origin/train_clustering_1.txt data/origin/train_clustering_2.txt data/origin/train_clustering_0.txt data/origin/train_clustering_4.txt' \
          --vdata 'data/origin/train_clustering_3.txt' \
		--test_data 'data/origin/Pub_remove_sim.txt data/origin/Chro_remove_sim.txt ./data/origin/test_clustering.txt' \
		--save_model 'trained_model/TADF_clustering_3.pt' \
		--vlabels 'Pubchem Chromophore TADF(test)' \
		--lr 1e-3 \
		--lr_decay 0.999 \
		--epochs 100000 \
		# --train
		#--test_data 'data/origin/LGD_DA_comb.txt data/origin/vis_chromophore.txt ./data/origin/test_clustering.txt' \

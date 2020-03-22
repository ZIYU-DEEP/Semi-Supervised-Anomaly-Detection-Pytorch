cd ..

python save_abnormal_xs.py --model forecast -nf ryerson_ab_train -af ryerson_ab_train_wn_1.4G
python save_abnormal_xs.py --model forecast -nf ryerson_ab_train -af ryerson_ab_train_fsk_1.4G
python save_abnormal_xs.py --model forecast -nf ryerson_ab_train -af ryerson_ab_train_psk_1.4G
python save_abnormal_xs.py --model forecast -nf ryerson_ab_train -af ryerson_ab_train_qam_1.4G
python save_abnormal_xs.py --model forecast -nf ryerson_ab_train -af ryerson_ab_train_ofdm_1.4G

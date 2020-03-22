cd ..

python save_abnormal_xs.py --model forecast -nf ryerson_ab_train -af ryerson_ab_train_wn_5G
python save_abnormal_xs.py --model forecast -nf ryerson_ab_train -af ryerson_ab_train_fsk_5G
python save_abnormal_xs.py --model forecast -nf ryerson_ab_train -af ryerson_ab_train_psk_5G
python save_abnormal_xs.py --model forecast -nf ryerson_ab_train -af ryerson_ab_train_qam_5G
python save_abnormal_xs.py --model forecast -nf ryerson_ab_train -af ryerson_ab_train_ofdm_5G

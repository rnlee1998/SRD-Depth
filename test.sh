for((i=12;i<20;i++))
do 
	CUDA_VISIBLE_DEVICES=1 nohup python -u evaluate_depth.py --load_weights_folder /mnt/data/liran/workdir/monovit/save_models/$*/mono_epc1_fa_ck_res50/models/weights_$i/ --eval_mono >>test_result_$*_epc1_fa_ck_res50.log 2>&1 &	
	sleep 30s   
	echo $i "test finish"
done






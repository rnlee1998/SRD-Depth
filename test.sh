for((i=12;i<20;i++))
do 
	CUDA_VISIBLE_DEVICES=9 nohup python -u evaluate_depth.py --load_weights_folder /mnt/data/liran/workdir/monovit/save_models/$*/mono_model/models/weights_$i/ --eval_mono >>test_result_$*.log 2>&1 &	
	sleep 25s   
	echo $i "test finish"
done


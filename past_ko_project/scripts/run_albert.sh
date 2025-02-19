for i in $(seq 1 12)
do
    python main.py --config configs/albert_covid19tweets.json --log_name albert_$i --train_datapath $i-2020
    for j in $(seq 1 12)
    do
        python test_checkpoint.py --config configs/albert_covid19tweets.json --checkpoint log/albert_$i/checkpoints/9.pt --test_dataset nela/$j-2020 --train_datapath $i-2020
    done
    python test_checkpoint.py --config configs/albert_covid19tweets.json --checkpoint log/albert_$i/checkpoints/9.pt --test_dataset midas --train_datapath $i-2020
done

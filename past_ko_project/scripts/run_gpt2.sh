for i in $(seq 1 12)
do
    python main.py --config configs/gpt2_covid19tweets.json --log_name gpt2_$i --train_datapath $i-2020
    for j in $(seq 1 12)
    do
        python test_checkpoint.py --config configs/gpt2_covid19tweets.json --checkpoint log/gpt2_$i/checkpoints/19.pt --test_dataset nela/$j-2020 --train_datapath $i-2020
    done
    python test_checkpoint.py --config configs/gpt2_covid19tweets.json --checkpoint log/gpt2_$i/checkpoints/19.pt --test_dataset midas --train_datapath $i-2020
done

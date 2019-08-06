python robonet/datasets/util/hdf5_2_records.py ~/hdf5 --robot sawyer --save_dir records_all_small/sawyer --n_workers 40;
python robonet/datasets/util/hdf5_2_records.py ~/hdf5 --robot R3 --save_dir record_all_small/R3 --n_workers 40;
python robonet/datasets/util/hdf5_2_records.py ~/hdf5 --robot widowx --save_dir records_all_small/widowx --n_workers 40;
python robonet/datasets/util/hdf5_2_records.py ~/hdf5 --robot baxter --save_dir record_all_small/baxter --n_workers 40;
python robonet/datasets/util/hdf5_2_records.py ~/hdf5 --robot fetch --save_dir records_all_small/fetch --n_workers 40;
python robonet/datasets/util/hdf5_2_records.py ~/hdf5 --robot franka --save_dir record_all_small/franka --n_workers 40;
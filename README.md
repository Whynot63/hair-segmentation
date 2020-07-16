# SETUP

Build and run docker:

```sh
docker build . --tag hair-segmentation
docker run --it hair-segmentation bash
```

# TRAIN

```sh
./prepare.sh
python3 train.py --data_dir dataset
```

# TEST

```
python3 test.py --selfie_dir ./test --output_dir ./test_output
```

# digikala-sentiment-lstm
> 🧠Trains a simple LSTM model on the Digikala product comment dataset for the sentiment classification task

![](sample.gif)


## Installation
Install `python` and `pip`. Create a `virtualenv` and activate it. Then:

```bash
$ git clone https://github.com/rajabzz/digikala-sentiment-lstm.git
$ cd digikala-sentiment-lstm
$ mkdir data
$ mkdir models
$ pip install -r requirements.txt
```
Copy your dataset to the `data` folder. If you don't have a dataset, consider using [digikala-crawler](https://github.com/rajabzz/digikala-crawler).

## Running The Program
The following command pre-processes the data, trains the LSTM model, evaluates it and starts an interactive mode for user's manual inputs:
```bash
$ python main.py
```
There exists some commandline arguments to make it even faster:
```bash
$ python main.py --full_data_path=data/results.jl
```
which uses the specified (full) path to read the raw data.
```bash
$ python main.py --training_data_ready --processed_pickle_data_path=processed_data.pickle
```
which uses the specified path to read the pre-processed data (from the previous step)
which is essentially the training data.
```bash
python main.py --data_model_ready --model_path=models/model.h5
```
which uses the specified path to read the saved model.
```bash
python main.py --max_length=128 --batch_size=20 --seed=42 --epoch=5
```
which tells the program to alter the defaults for maximum comment length,
batch size for model training, seed for the random object and number of epochs
for model training respectively. The specified values are actually the default
values.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Task List
- [ ] Split the code into multiple files.
- [x] <del>Use `sys.argv` instead of manually changing the variables inside the code.</del>
README

Requirements:

To use this script you need to install first a couple of libraries by running these command from a terminal shell.

- pip install keras==2.1.2

(pip uninstall keras first if you already have another version of keras)

- pip install spacy

- python -m spacy download en_vectors_web_lg

(to download the embedding vectors dictionary)

If you will get a system message telling you that your laptop is running out of memory 
you just need to restart the python kernel to remove all the cache from the RAM (or restart the whole laptop).

NB: 
- the script works only with the SemEval datasets for now, I will soon generalize the code to let the user set the training and target columns of whatever dataset.

- the code actually return the error 'SystemExit' (An exception has occurred, use %tb to see the full traceback.). This is actually anerror message returned by tensorflow for unknown reasons since it doesn't really stop the script (that work till the end and save everything). I will try to figure out how to silent this error.

# Assignment 2 Machine Learning and Data Mining - Handwriting Recognition
Project for assignment 1 of Machine Learning and Data Mining course

This project has 5 main files:

1. Handwriting.py - Contains the main class for prediction, it also has embededd load file and cross validation features

2. Main.py - Main file/class that can depending on the parameters passed 
    - Run all experiments ( 8 algorithms with different Parameters ) in multiprocessing way
    - Run all experiments in sequential way
    - Run the two best Algorithms with 3 PCA variations
    - Display the first 100 samples of the dataset as an image (16x16 Matrix)

3. results.log: this is the file generated after executing the code with meaningful information about the execution
timeline

4. report: folder with latex files to create the report

5. semeion.data: file contained 1593 samples of hadwriting digits with 256 features each corresponding to a 16x16 matrix

## How to run the code

In order to run the code you go to the root folder and run

```bash
python Main.py
```
This will run the default mode of the project. Which is running the 2 best algorithms with 3 variations of PCA

If you want to use other modes you can run the project with parameters:

 ```bash
python Main.py [all_multi | all_seq | best | image]
 ```

This will run the algorithm in the one of the modes explained previously
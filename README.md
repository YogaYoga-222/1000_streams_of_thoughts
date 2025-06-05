# Stream of Thoughts Similarity Checker

This project checks if a new thought is similar to any existing thought using **TF-IDF** and **cosine similarity**. It helps to find out whether the new text is already known or different.

## What it does

- Loads and cleans a CSV file with many thoughts
- Combines important text columns into one
- Splits the data into training and testing sets
- Converts text into numbers using TF-IDF
- Compares each test thought with the training data
- Tells if the test thought is similar or new/different

## Dataset

You need a file named:  
`stream_of_thoughts.csv`  

The file should have these columns:
- `Raw`, `Text`, `Text.1`, `Unnamed: 4`, `Unnamed: 5`

These columns will be joined into one line for each thought.

## 🛠️ Requirements

Install required Python libraries:

```bash
pip install pandas scikit-learn
```
## Notes

* TF-IDF is used to change text into numbers.

* Cosine similarity helps compare how close two texts are.

* random_state=42 makes sure the sample stays the same every time you run.

## Example Output

```bash
Test Thought 9: Stream of 8 for Young
Most similar training thought: Stream of 34 for Young
Similarity Score: 0.73
This is a known or similar thought.
```
## Run the Script
```bash
python3 1000_streams_of_thoughts
```




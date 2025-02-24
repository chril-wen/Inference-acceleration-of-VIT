from datasets import load_dataset

ds = load_dataset("albertvillanova/medmnist-v2", "pathmnist",data_dir="./image")    # download & load the dataset

if __name__ == "__main__":
    labels = ds["train"]["label"]    # get the labels
    print(labels)    # print the labels


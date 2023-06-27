import argparse
import os
import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, help="Path to dataset to be used as input")
    parser.add_argument('--output_path', required=True, help="Path to output directory")
    args = parser.parse_args()

    # Creates needed directories when necessary
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if not os.path.exists(os.path.join(args.output_path, "csv")):
        os.makedirs(os.path.join(args.output_path, "csv"))

    if not os.path.exists(os.path.join(args.output_path, "plots")):
        os.makedirs(os.path.join(args.output_path, "plots"))


    df, df_min = utils.generateDataFrame(args.dataset_path, os.path.join(args.output_path, "csv"))
    # print(df.describe())
    print(df_min.head())
    utils.generatePlots(df, os.path.join(args.output_path, "plots"))

if __name__ == "__main__":
    main()
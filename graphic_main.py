import argparse
import os
import pandas as pd
import graphic_utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_path_complete', required=True, help="Path to dataset (complete_analysis) to be used as input")
    parser.add_argument('--ds_path_summed', required=True, help="Path to dataset (summed_analysis) to be used as input")
    parser.add_argument('--output_path', required=True, help="Path to output directory")
    args = parser.parse_args()

    # Creates needed directories when necessary
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    df_complete = pd.read_pickle(args.ds_path_complete)
    df_summed = pd.read_csv(args.ds_path_summed)
    output_path = args.output_path

    graphic_utils.hist_vr(df_summed,output_path)
    graphic_utils.hist_bpm(df_complete,output_path)
    graphic_utils.hist_distribution_F0(df_complete,output_path)
    graphic_utils.distribution_of_pitch_gender(df_complete,output_path)
    graphic_utils.hist_mean_F0(df_complete,output_path)

if __name__ == "__main__":
    main()
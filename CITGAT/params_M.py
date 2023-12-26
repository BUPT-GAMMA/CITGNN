import argparse
import sys

argv = sys.argv
dataset = argv[1]


def cora_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="cora")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--p', type=float, default="0.02")
    parser.add_argument('--epochtimes', type=int, default=20)  
    parser.add_argument('--clusters', type=int, default=100) 
    parser.add_argument('--cuda_id', type=int, default=0)
    args, _ = parser.parse_known_args()
    return args


def citeseer_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="citeseer")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--p', type=float, default="0.02")
    parser.add_argument('--epochtimes', type=int, default=20)  
    parser.add_argument('--clusters', type=int, default=100) 
    parser.add_argument('--cuda_id', type=int, default=0)
    args, _ = parser.parse_known_args()
    return args

def pubmed_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="pubmed")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--p', type=float, default="0.02")
    parser.add_argument('--epochtimes', type=int, default=20)  
    parser.add_argument('--clusters', type=int, default=100) 
    parser.add_argument('--cuda_id', type=int, default=0)
    args, _ = parser.parse_known_args()
    return args

def set_params():
    if dataset == "cora":
        args = cora_params()
    elif dataset == "citeseer":
        args = citeseer_params()
    elif dataset == "pubmed":
        args = pubmed_params()

    return args
    
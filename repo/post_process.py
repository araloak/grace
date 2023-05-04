import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input', default="", type=str, required=False)
parser.add_argument('--output', default="", type=str, required=False)
args = parser.parse_args()

file = open(args.input,encoding="utf8").readlines()
with open(args.output,"w",encoding="utf8") as f:
    for line in file:
        line = line.strip().replace("#<|endoftext|>","")
        if line!="":
            f.write(line+"\n")
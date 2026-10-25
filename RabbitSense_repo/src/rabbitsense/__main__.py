import argparse
from . import v36, v37

def main():
    parser=argparse.ArgumentParser(description="RabbitSense: análisis experimental de movimiento")
    parser.add_argument("--version",choices=["v36","v37"],default="v36",help="v36 referencia; v37 experimental")
    parser.add_argument("--video",required=True)
    parser.add_argument("--tracks",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--step",type=float,default=.5)
    parser.add_argument("--threshold",type=float,default=.2)
    parser.add_argument("--max-jump",type=float,default=1.5)
    parser.add_argument("--max-scale-change",type=float,default=1.8)
    parser.add_argument("--video-out",action="store_true")
    args=parser.parse_args()
    (v36 if args.version=="v36" else v37).run(args)

if __name__=="__main__": main()

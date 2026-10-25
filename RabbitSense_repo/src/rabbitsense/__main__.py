import argparse
import math
from . import motion, v36, v37

def main():
    parser=argparse.ArgumentParser(description="RabbitSense: análisis experimental de movimiento")
    parser.add_argument("--version",choices=["flow","v36","v37"],default="v36",help="v36 referencia por defecto; flow y v37 experimentales")
    parser.add_argument("--video",required=True)
    parser.add_argument("--tracks",help="CSV interpolado para referencias v36/v37")
    parser.add_argument("--anchors",help="JSON original de anclas para flow")
    parser.add_argument("--out",required=True)
    parser.add_argument("--step",type=float,default=.5)
    parser.add_argument("--threshold",type=float,default=.2)
    parser.add_argument("--max-jump",type=float,default=1.5)
    parser.add_argument("--max-scale-change",type=float,default=1.8)
    parser.add_argument("--video-out",action="store_true")
    args=parser.parse_args()
    if not math.isfinite(args.step) or not math.isfinite(args.threshold) or args.step <= 0 or args.threshold < 0:
        parser.error("--step debe ser positivo; --threshold no negativo")
    if args.version == "flow" and not args.anchors:
        parser.error("flow requiere --anchors")
    if args.version != "flow" and not args.tracks:
        parser.error("v36/v37 requieren --tracks")
    {"flow": motion, "v36": v36, "v37": v37}[args.version].run(args)

if __name__=="__main__": main()

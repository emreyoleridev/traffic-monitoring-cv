"""Traffic monitoring: vehicle detection, counting and congestion analytics on road-camera images."""

# Our 3-class taxonomy. The raw RF100 "vehicles" set mixes two labelling schemes
# (e.g. "big truck" vs "truck-l-"), so everything is folded into car / bus / truck.
CLASSES = ["car", "bus", "truck"]
RAW_TO_OURS = {
    "car": 0,
    "big bus": 1, "small bus": 1, "bus-l-": 1, "bus-s-": 1,
    "big truck": 2, "mid truck": 2, "small truck": 2,
    "truck-s-": 2, "truck-m-": 2, "truck-l-": 2, "truck-xl-": 2,
}
# COCO ids of the same concepts, used for the zero-shot (pretrained) baseline.
COCO_TO_OURS = {2: 0, 5: 1, 7: 2}

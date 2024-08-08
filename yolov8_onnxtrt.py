import argparse
from ultralytics import YOLO

def export_model(weights, input_size):
    # Initialize the model with the provided weights file
    model = YOLO(weights)
    inp_size_vals = str.split(input_size, "x")
    
    # Export the model to ONNX format with TensorRT optimization
    model.export(format="onnx_trt", imgsz=(int(inp_size_vals[1]), int(inp_size_vals[0])), dynamic=False) #imgsz=(512, 1024)

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX with TensorRT optimization.")
    
    # Add the -w/--weights argument to specify the weights file
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to the YOLO weights file (e.g., yolov8n.pt)')
    parser.add_argument('-s', '--input_size', type=str, required=True, help="Input tensor size in format WxH. For example: 1024x512")
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the export_model function with the provided weights file
    export_model(args.weights, args.input_size)

if __name__ == "__main__":
    main()


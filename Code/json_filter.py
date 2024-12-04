import json

def replace_bboxes_with_value(json_file, output_file, width_value=10, height_value=20):
    # Read the JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Replace bounding boxes with specific width and height, and update area/iscrowd
    for annotation in data["annotations"]:
        if "bbox" in annotation:
            # Replace each bounding box with specific width and height
            x, y = annotation["bbox"][:2]  # Keep the original x, y (if applicable)
            annotation["bbox"] = [x, y, width_value, height_value]
            
            # Calculate the area as width * height
            annotation["area"] = width_value * height_value
            
            # Set iscrowd to 1 if the area is greater than 0, otherwise 0
            annotation["iscrowd"] = 1 if annotation["area"] > 0 else 0

    # Write the modified data to the output file
    with open(output_file, 'w') as output_file:
        json.dump(data, output_file, indent=4)

    print(f"Updated JSON saved to {output_file}")

# Example usage:
replace_bboxes_with_value("/home/ai-13/Desktop/Project/GDINO_Project_BILAL_data_T1/annotations/test.json", "/home/ai-13/Desktop/Project/GDINO_Project_BILAL_data_T1/annotations/test_new.json", width_value=10, height_value=20)

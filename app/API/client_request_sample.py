import requests
import argparse

def predict_age_and_match(image_path_1, image_path_2, api_url="http://localhost:8000/v1/infer_age_and_match"):
    """
    Sends two images to the age and face matching API and returns the response.

    Parameters
    ----------
    image_path_1 : str
        Path to the first image file.
    image_path_2 : str
        Path to the second image file.
    api_url : str
        URL of the API endpoint.

    Returns
    -------
    dict
        The JSON response from the API containing age and match results.
    """
    with open(image_path_1, 'rb') as img1, open(image_path_2, 'rb') as img2:
        files = {
            'image_file_1': img1,
            'image_file_2': img2,
        }
        response = requests.post(api_url, files=files)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    
def parse_args():
    parser = argparse.ArgumentParser(description="Client for Age and Face Matching API")
    parser.add_argument("--image1", type=str, help="Path to the first image file")
    parser.add_argument("--image2", type=str, help="Path to the second image file")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000/v1/infer_age_and_match", help="API endpoint URL")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    results = predict_age_and_match(args.image1, args.image2, args.api_url)
    print("Inference Results:")
    print(results)

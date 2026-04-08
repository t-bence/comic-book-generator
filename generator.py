import io
import os
from typing import List

from dotenv import load_dotenv
from google import genai
from PIL import Image

# Load environment variables from .env file
load_dotenv()


def _ensure_directory_exists(file_path: str):
    """Ensures the directory for the given file path exists."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def _call_gemini_image_model(contents: List, output_path: str) -> str:
    """
    Helper function to call Gemini model and extract/save the image.

    Args:
        contents (List): Multimodal contents (strings, images, etc.).
        output_path (str): Path to save the resulting image.

    Returns:
        str: Absolute path of the saved image.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables. Please set it in your .env file."
        )

    client = genai.Client(api_key=api_key)

    try:
        # Call the Gemini model capable of image output
        response = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=contents,
        )

        if (
            not response.candidates
            or not response.candidates[0].content
            or not response.candidates[0].content.parts
        ):
            # Enhanced logging for debugging
            print(f"DEBUG: Response object: {response}")

            error_msg = "No parts found in the response from the model."
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.finish_reason:
                    error_msg += f" Finish reason: {candidate.finish_reason}"
                    error_msg += f"Content: {candidate.content}"
                if candidate.safety_ratings:
                    blocked_ratings = [r for r in candidate.safety_ratings if r.blocked]
                    if blocked_ratings:
                        error_msg += f" Safety ratings: {blocked_ratings}"

            raise ValueError(error_msg)

        image_saved = False
        for part in response.candidates[0].content.parts:
            # Inline data contains image bytes if the model generated an image
            if part.inline_data:
                image_bytes = part.inline_data.data
                image = Image.open(io.BytesIO(image_bytes))
                _ensure_directory_exists(output_path)
                image.save(output_path)
                image_saved = True
                break
            # Some versions of the SDK might use as_image() if available
            elif hasattr(part, "as_image"):
                image = part.as_image()
                if image:
                    _ensure_directory_exists(output_path)
                    image.save(output_path)
                    image_saved = True
                    break

        if not image_saved:
            # If no image was found, check for text (might be a safety filter rejection or explanation)
            text_response = " ".join(
                [
                    part.text
                    for part in response.candidates[0].content.parts
                    if part.text
                ]
            )
            raise ValueError(
                f"No image was generated. Model response text: {text_response}"
            )

        abs_path = os.path.abspath(output_path)
        print(f"Image successfully saved to: {abs_path}")
        return abs_path

    except Exception as e:
        print(f"Error calling Gemini model: {e}")
        raise


def generate_initial_image(description: str, output_path: str) -> str:
    """
    Generates an initial image from a text prompt.

    Args:
        prompt (str): The text description of the image.
        output_path (str): The path to save the generated image.

    Returns:
        str: Path of the saved image.
    """

    prompt = (
        "Create one page of a comic book based on the description. "
        f"Create 4 images arranged on one page. Here's the description: {description}"
    )

    print("Generating initial image")
    return _call_gemini_image_model([prompt], output_path)


def generate_next_comic_panel(previous_image_path: str, output_path: str) -> str:
    """
    Generates the next comic panel based on a previous image.

    Args:
        previous_image_path (str): Path to the preceding image.
        output_path (str): Path to save the next panel.

    Returns:
        str: Path of the saved image.
    """
    if not os.path.exists(previous_image_path):
        raise FileNotFoundError(f"Previous image not found at: {previous_image_path}")

    print(f"Generating comic panel {output_path}")

    previous_image = Image.open(previous_image_path)
    prompt = (
        "You get an image, a page from a comic book. "
        "Generate the next page in the comic book to continue the story."
    )

    return _call_gemini_image_model([prompt, previous_image], output_path)

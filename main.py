import argparse
import os
import uuid

from generator import generate_initial_image, generate_next_comic_panel


def run_comic_sequence(initial_prompt: str, num_images: int = 3):
    """
    Orchestrates the generation of a comic book sequence.

    Args:
        initial_prompt (str): The prompt for the first panel.
        num_images (int): Number of panels to generate.
    """
    # Create a unique run ID and directory
    run_id = str(uuid.uuid4())
    run_dir = os.path.join("comics", run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Starting new comic sequence run: {run_id}")
    print(f"Images will be saved in: {run_dir}\n")

    try:
        # 1. Generate the first image
        panel_1_name = "001.png"
        panel_1_path = os.path.join(run_dir, panel_1_name)
        current_panel_path = generate_initial_image(initial_prompt, panel_1_path)

        # 2. Generate subsequent panels
        for i in range(2, num_images + 1):
            next_panel_name = f"{i:03d}.png"
            next_panel_path = os.path.join(run_dir, next_panel_name)

            # Pass the previous panel's path to generate the next one
            current_panel_path = generate_next_comic_panel(
                current_panel_path, next_panel_path
            )

        print(f"\nSuccessfully generated {num_images} panels in: {run_dir}")

    except Exception as e:
        print(f"\nAn error occurred during the comic sequence generation: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate a comic book sequence.")
    parser.add_argument(
        "--prompt",
        type=str,
        help="The description to use for the first image.",
        default="A baby is discovering its home, playing with books and putting toys from here to there and then again somewhere else.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        help="Number of images to generate.",
        default=3,
    )
    args = parser.parse_args()

    run_comic_sequence(args.prompt, args.num_images)


if __name__ == "__main__":
    main()

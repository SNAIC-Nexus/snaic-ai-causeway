import argparse
import os
import cv2

# List to store the clicked points
clicked_points = []


def mouse_click_callback(event, x, y, flags, param):
    """Tracks mouse clicks and draws the polygon points in real-time."""
    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Added point: ({x}, {y})")

        # Draw a small circle at the click location
        cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1)

        # Draw a line connecting to the previous point
        if len(clicked_points) > 1:
            cv2.line(
                display_image,
                clicked_points[-2],
                clicked_points[-1],
                (0, 255, 255),
                2,
            )

        cv2.imshow("Coordinate Picker", display_image)


if __name__ == "__main__":
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Pick coordinate boundaries on an LTA traffic image."
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        required=True,
        help="Path to the LTA traffic image file.",
    )
    args = parser.parse_args()

    # 2. Check if file exists and load it
    if not os.path.exists(args.image):
        print(f"Error: The file '{args.image}' does not exist.")
        exit(1)

    image = cv2.imread(args.image)
    if image is None:
        print(
            f"Error: Could not decode the image file '{args.image}'. Is it a valid image?"
        )
        exit(1)

    # Clone the image for real-time drawing feedback
    display_image = image.copy()

    print(f"=== LTA Camera Coordinate Picker ===")
    print(f"Loaded image: {args.image}")
    print("Instructions:")
    print("1. LEFT-CLICK to select the corners of a lane sequentially.")
    print("2. Look at your terminal to see the printed (x, y) tuples.")
    print("3. Press 'c' to CLEAR your current points and restart.")
    print("4. Press 'q' or 'ESC' to QUIT when finished.")
    print("-" * 36)

    # 3. Create window and bind mouse events
    cv2.namedWindow("Coordinate Picker")
    cv2.setMouseCallback("Coordinate Picker", mouse_click_callback)
    cv2.imshow("Coordinate Picker", display_image)

    while True:
        key = cv2.waitKey(1) & 0xFF

        # Press 'c' to clear and reset points
        if key == ord("c"):
            clicked_points = []
            display_image = image.copy()
            cv2.imshow("Coordinate Picker", display_image)
            print("\nCleared points. Start clicking again.")

        # Press 'q' or ESC to exit
        elif key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()

    # Output the final list format
    print("\n" + "=" * 36)
    print("FINAL POLYGON COORDINATES:")
    print("=" * 36)
    print(clicked_points)
    print("=" * 36)

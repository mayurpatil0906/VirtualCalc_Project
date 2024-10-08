import cv2
import numpy as np
import mediapipe as mp
import pytesseract
import sympy as sp
from matplotlib import pyplot as plt

# Initialize Mediapipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.9)

# Initialize drawing canvas and undo/redo stacks
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
undo_stack = []
redo_stack = []

# Last position for drawing
last_x, last_y = None, None

# Variables to store recognized equation and solution
recognized_equation = ""
solution = ""

# Function to process the frame and detect hand landmarks
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    return results

# Function to extract the positions of the fingers
def extract_landmarks(results):
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.append((int(lm.x * 640), int(lm.y * 480)))  # Scale to frame size
        return landmarks
    return None

# Function to enhance the video quality
def enhance_video(frame):
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 30    # Brightness control (0-100)
    enhanced_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    enhanced_frame = cv2.GaussianBlur(enhanced_frame, (5, 5), 0)
    return enhanced_frame

# Function to recognize and solve the drawn equation
def recognize_and_solve_equation(canvas):
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_canvas, 150, 255, cv2.THRESH_BINARY_INV)
    equation = pytesseract.image_to_string(thresh, config='--psm 6').strip()  # Clean up the extracted string

    if equation:
        try:
            x = sp.symbols('x')
            solution = sp.solve(equation, x)
            return equation, solution
        except Exception as e:
            return equation, f"Error solving: {e}"

    return None, None

# Function to draw buttons on the frame
def draw_buttons(frame):
    buttons = [
        {"label": "Undo", "pos": (10, 50)},
        {"label": "Redo", "pos": (110, 50)},
        {"label": "Enter", "pos": (210, 50)},
        {"label": "Select", "pos": (310, 50)},
    ]
    for button in buttons:
        cv2.rectangle(frame, button['pos'], (button['pos'][0] + 80, button['pos'][1] + 40), (255, 0, 0), -1)
        cv2.putText(frame, button['label'], (button['pos'][0] + 10, button['pos'][1] + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

# Function to check button click
def check_button_click(cursor_pos):
    button_positions = {
        "Undo": (10, 50, 90, 90),
        "Redo": (110, 50, 190, 90),
        "Enter": (210, 50, 290, 90),
        "Select": (310, 50, 390, 90),
    }
    for button, (x1, y1, x2, y2) in button_positions.items():
        if x1 <= cursor_pos[0] <= x2 and y1 <= cursor_pos[1] <= y2:
            return button
    return None

# Function to display the frame using Matplotlib
def display_frame(frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.draw()
    plt.pause(0.001)  # Short pause to update the display
    plt.clf()  # Clear the figure for the next frame

# Function to smooth cursor position
def smooth_position(prev_pos, curr_pos, alpha=0.2):
    """Smooth the cursor position using an exponential moving average."""
    if prev_pos[0] is None or prev_pos[1] is None:
        return curr_pos  # Return current position if no previous position exists
    
    smooth_x = (1 - alpha) * prev_pos[0] + alpha * curr_pos[0]
    smooth_y = (1 - alpha) * prev_pos[1] + alpha * curr_pos[1]
    
    return smooth_x, smooth_y

# Main function for virtual mouse and drawing
def virtual_mouse_drawing():
    global last_x, last_y, recognized_equation, solution, canvas, undo_stack, redo_stack
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame horizontally to remove mirror effect
        frame = cv2.flip(frame, 1)

        # Enhance video quality
        enhanced_frame = enhance_video(frame)

        # Process the frame to detect hand landmarks
        results = process_frame(enhanced_frame)
        landmarks = extract_landmarks(results)

        if landmarks:
            # Get the tip positions for index and middle fingers
            index_finger = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
            middle_finger = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value]

            # Smooth cursor position
            cursor_x, cursor_y = index_finger
            cursor_x, cursor_y = smooth_position((last_x, last_y), (cursor_x, cursor_y), alpha=0.2)
            cv2.circle(enhanced_frame, (int(cursor_x), int(cursor_y)), 10, (0, 255, 0), -1)  # Draw cursor

            # Check for button clicks
            clicked_button = check_button_click((cursor_x, cursor_y))
            if clicked_button == "Undo" and undo_stack:
                redo_stack.append(canvas.copy())
                canvas = undo_stack.pop()
            elif clicked_button == "Redo" and redo_stack:
                undo_stack.append(canvas.copy())
                canvas = redo_stack.pop()
            elif clicked_button == "Enter":
                recognized_equation, solution = recognize_and_solve_equation(canvas)

            # Determine if we are drawing or ignoring based on the middle finger's position
            if middle_finger[1] < cursor_y:  # Middle finger above index finger
                if last_x is not None and last_y is not None:
                    # Check if we're clicking a button or drawing
                    if not clicked_button:
                        undo_stack.append(canvas.copy())  # Save current state for undo
                        cv2.line(canvas, (int(last_x), int(last_y)), (int(cursor_x), int(cursor_y)), (255, 255, 255), 5)
                last_x, last_y = cursor_x, cursor_y  # Update last position for drawing
            else:
                last_x, last_y = None, None  # Reset last position to stop drawing

        # Draw buttons on the frame
        enhanced_frame = draw_buttons(enhanced_frame)

        # Display the canvas
        combined_frame = cv2.addWeighted(enhanced_frame, 0.5, canvas, 0.5, 0)

        # Display recognized equation and its solution on the frame
        if recognized_equation:
            cv2.putText(combined_frame, f"Equation: {recognized_equation}", (10, 460), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined_frame, f"Solution: {solution}", (10, 480), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the combined frame using Matplotlib
        display_frame(combined_frame)

        # Check for quit condition
        if plt.waitforbuttonpress(0.01):  # Wait for a short time to check for keypress
            break

    cap.release()
    plt.close()  # Close the matplotlib window
    cv2.destroyAllWindows()

# Start the virtual mouse and drawing functionality
if __name__ == "__main__":
    virtual_mouse_drawing()

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import simpledialog
import time
import threading

class FocusTimerApp:
    def __init__(self):
        # --- Configuration ---
        self.CONFIDENCE_THRESHOLD = 0.5
        # Sensitivity: Higher = harder to trigger distraction, Lower = sensitive
        # Range 0.0 to 1.0. 0.5 is center.
        # 0.70 means you have to look significantly away to trigger it.
        self.HORIZONTAL_RATIO_THRESHOLD = 0.70 
        
        # --- State Variables ---
        self.is_focused = False
        self.time_remaining = 0
        self.running = True
        self.video_running = True

        # --- Setup MediaPipe Face Mesh ---
        # We use refine_landmarks=True to get specific Iris points
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, 
            min_detection_confidence=self.CONFIDENCE_THRESHOLD,
            min_tracking_confidence=self.CONFIDENCE_THRESHOLD
        )

        # --- Setup GUI ---
        self.root = tk.Tk()
        self.root.withdraw() # Hide the main root window

        # 1. Get User Input for Duration
        minutes = simpledialog.askinteger("Focus Mode", "Enter focus duration (minutes):", minvalue=1, maxvalue=180)
        if minutes is None:
            self.running = False
            self.root.destroy()
            return
        
        self.time_remaining = minutes * 60

        # 2. Setup Overlay Window (The Red Tint)
        self.overlay = tk.Toplevel(self.root)
        self.overlay.attributes("-fullscreen", True)
        self.overlay.attributes("-topmost", True)
        self.overlay.attributes("-alpha", 0.0) # Start transparent
        self.overlay.configure(bg='red')
        # Attempt to make the red color clickable-through (Windows specific optimization)
        try:
            self.overlay.attributes("-transparentcolor", "white") 
        except:
            pass 

        # 3. Setup Timer Window (Small floating widget)
        self.timer_window = tk.Toplevel(self.root)
        self.timer_window.geometry("200x80+50+50") # Top left corner placement
        self.timer_window.overrideredirect(True) # Remove title bar
        self.timer_window.attributes("-topmost", True)
        self.timer_window.configure(bg='black')

        self.timer_label = tk.Label(
            self.timer_window, 
            text="00:00", 
            font=("Arial", 30, "bold"), 
            fg="green", 
            bg="black"
        )
        self.timer_label.pack(expand=True)

        # --- Start Processes ---
        self.cap = cv2.VideoCapture(0)
        
        # Start the timer logic in a separate thread so it doesn't block the video processing
        self.timer_thread = threading.Thread(target=self.run_timer)
        self.timer_thread.daemon = True
        self.timer_thread.start()

        # Start the video processing loop
        self.process_video()
        self.root.mainloop()

    def get_iris_position(self, iris_center, right_point, left_point):
        """
        Calculates the ratio of the iris position relative to eye corners.
        Returns a value between 0.0 and 1.0.
        ~0.5 is perfectly centered.
        """
        center_to_right_dist = np.linalg.norm(iris_center - right_point)
        total_distance = np.linalg.norm(right_point - left_point)
        ratio = center_to_right_dist / total_distance
        return ratio

    def process_video(self):
        if not self.video_running:
            return

        success, image = self.cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            self.root.after(10, self.process_video)
            return

        # Flip image for mirror effect
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        img_h, img_w = image.shape[:2]
        currently_focused = False

        if results.multi_face_landmarks:
            # Convert normalized landmarks to pixel coordinates
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            # --- MediaPipe Face Mesh Indices ---
            # These specific indices define the eye corners and the iris
            
            # Left Eye
            LEFT_IRIS = [474, 475, 476, 477]
            L_H_LEFT = 33  # Left eye left corner
            L_H_RIGHT = 133 # Left eye right corner
            
            # Right Eye
            RIGHT_IRIS = [469, 470, 471, 472]
            R_H_LEFT = 362 # Right eye left corner
            R_H_RIGHT = 263 # Right eye right corner

            try:
                #  
                # Note: We use the specific landmark indices above to isolate the eyes.

                # Left Eye Calculation
                l_iris_center = np.mean(mesh_points[LEFT_IRIS], axis=0)
                l_right_p = mesh_points[L_H_RIGHT]
                l_left_p = mesh_points[L_H_LEFT]
                l_ratio = self.get_iris_position(l_iris_center, l_right_p, l_left_p)

                # Right Eye Calculation
                r_iris_center = np.mean(mesh_points[RIGHT_IRIS], axis=0)
                r_right_p = mesh_points[R_H_RIGHT]
                r_left_p = mesh_points[R_H_LEFT]
                r_ratio = self.get_iris_position(r_iris_center, r_right_p, r_left_p)

                # Average Ratio
                gaze_ratio = (l_ratio + r_ratio) / 2

                # 
                # Note: If ratio > 0.7 or < 0.3 (approx), the user is looking away.

                # Determine Focus based on thresholds
                # If gaze is within the accepted range, user is "Focused"
                if (1 - self.HORIZONTAL_RATIO_THRESHOLD) < gaze_ratio < self.HORIZONTAL_RATIO_THRESHOLD:
                    currently_focused = True
                else:
                    currently_focused = False
            except Exception as e:
                # If math fails (e.g., blink or lost tracking), assume distracted to be safe
                currently_focused = False
        else:
            # No face detected (Looking away from camera completely)
            currently_focused = False

        self.update_ui_state(currently_focused)
        
        # Repeat after 30ms (approx 30 FPS)
        self.root.after(30, self.process_video)

    def update_ui_state(self, focused):
        self.is_focused = focused
        
        if self.is_focused:
            # User is focused: Transparent overlay, Green Timer
            self.overlay.attributes("-alpha", 0.0) 
            self.timer_label.config(fg="#00ff00") # Green text
        else:
            # User distracted: Red Overlay, Red Timer
            self.overlay.attributes("-alpha", 0.3) # 30% opacity red
            self.timer_label.config(fg="#ff0000") # Red text

    def run_timer(self):
        while self.time_remaining > 0 and self.running:
            # Only count down if user is focused
            if self.is_focused:
                mins, secs = divmod(self.time_remaining, 60)
                time_format = '{:02d}:{:02d}'.format(mins, secs)
                
                try:
                    self.timer_label.config(text=time_format)
                except:
                    break

                time.sleep(1)
                self.time_remaining -= 1
            else:
                # If distracted, pause (do not decrement time) and wait briefly
                time.sleep(0.1)
        
        # Timer Finished
        if self.running:
            self.finish_session()

    def finish_session(self):
        self.video_running = False
        self.cap.release()
        try:
            self.timer_label.config(text="DONE!", fg="white")
            self.overlay.attributes("-alpha", 0.0)
            simpledialog.messagebox.showinfo("Focus Mode", "Session Complete!")
            self.root.quit()
        except:
            pass

if __name__ == "__main__":
    try:
        app = FocusTimerApp()
    except KeyboardInterrupt:
        print("Program stopped.")
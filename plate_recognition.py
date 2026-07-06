import cv2
import numpy as np
import easyocr
import os

def main():
    image_path = 'car.png'
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found. Please make sure the image is in the same directory.")
        return

    print("Loading original image...")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    # --- Step 1: Grayscale & Blur ---
    print("\n--- Step 1: Converting to Grayscale & Applying Blur ---")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter is excellent for noise removal while keeping edges sharp
    gray_blur = cv2.bilateralFilter(gray, 11, 17, 17)
    
    cv2.imwrite('step1_gray_blur.png', gray_blur)
    print("Pre-processed image saved as 'step1_gray_blur.png'.")
    input(">>> STAGE 1: Take a screenshot of 'step1_gray_blur.png' in your folder. Then press Enter to continue to Stage 2...")

    # --- Step 2: Edge Detection ---
    print("\n--- Step 2: Applying Canny Edge Detection ---")
    # Detect edges
    edges = cv2.Canny(gray_blur, 30, 200)
    
    cv2.imwrite('step2_edges.png', edges)
    print("Edge-detected image saved as 'step2_edges.png'.")
    input(">>> STAGE 2: Take a screenshot of 'step2_edges.png' in your folder. Then press Enter to continue to Stage 3...")

    # --- Step 3: Finding Contours and Isolating the Plate ---
    print("\n--- Step 3: Finding Contours and Isolating the Plate ---")
    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area in descending order and keep the top 30
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    
    plate_contour = None
    x, y, w, h = 0, 0, 0, 0
    
    for c in contours:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        
        # A license plate is a rectangle, so it has 4 vertices
        if len(approx) == 4:
            plate_contour = approx
            x, y, w, h = cv2.boundingRect(c)
            break

    if plate_contour is None:
        print("Warning: Could not find a 4-point contour. Falling back to the largest rectangular-like contour...")
        # Fallback: find the largest contour that looks like a rectangle
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            # License plates are wider than they are tall (usually ratio between 2 and 5)
            if 2.0 <= aspect_ratio <= 5.5:
                plate_contour = c
                break

    if plate_contour is not None:
        # Draw the detected contour in green on a copy of the original image
        contour_img = img.copy()
        cv2.drawContours(contour_img, [plate_contour], -1, (0, 255, 0), 3)
        cv2.imwrite('step3_contours.png', contour_img)
        print("Image with plate contour outlined saved as 'step3_contours.png'.")
        
        # Crop the plate region
        cropped_plate = img[y:y+h, x:x+w]
        cv2.imwrite('step4_cropped_plate.png', cropped_plate)
        print("Cropped license plate saved as 'step4_cropped_plate.png'.")
    else:
        print("Error: Could not isolate any license plate contour.")
        return

    input(">>> STAGE 3: Take screenshots of 'step3_contours.png' and 'step4_cropped_plate.png'. Then press Enter to continue to Stage 4 (OCR)...")

    # --- Step 4: Optical Character Recognition (OCR) ---
    print("\n--- Step 4: Running OCR on the Cropped Plate ---")
    print("Initializing EasyOCR reader (English & Turkish language packs)...")
    reader = easyocr.Reader(['en', 'tr'])
    
    print("Reading text from cropped plate...")
    results = reader.readtext(cropped_plate)
    
    print("\n==========================================")
    print("OCR Results:")
    print("==========================================")
    if not results:
        print("No text detected by OCR.")
    else:
        for (bbox, text, prob) in results:
            print(f"Detected Text: {text} (Confidence: {prob:.2f})")
            
    print("==========================================\n")
    print("ALPR Pipeline complete!")

if __name__ == '__main__':
    main()

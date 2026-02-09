import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing import image

print("=" * 70)
print(" Testing the model on new images")
print("=" * 70)


MODEL_PATH = r'C:\Users\AliKhalid\PycharmProjects\PythonProject\best_model_v2.h5'
IMG_SIZE = (150, 150)

# 1. Load Model
# ============================================================================

print("\n Loading the model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f" Model loaded from: {MODEL_PATH}")
except Exception as e:
    print(f" Error: {e}")
    exit()



# 2. Predict on a single image
# ============================================================================

def predict_image(img_path, show_image=True):
    """
    Predict on a single image with visualization
    """
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediction
        prediction = model.predict(img_array, verbose=0)
        prob = prediction[0][0]

        # Interpret result
        if prob > 0.5:
            diagnosis = "🔴 Pneumonia"
            confidence = prob
            color = 'red'
        else:
            diagnosis = "🟢 Normal"
            confidence = 1 - prob
            color = 'green'

        # Print result
        print(f"\n Image: {os.path.basename(img_path)}")
        print(f"   Diagnosis: {diagnosis}")
        print(f"   Confidence: {confidence * 100:.2f}%")
        print(f"   Pneumonia probability: {prob * 100:.2f}%")
        print(f"   Normal probability: {(1 - prob) * 100:.2f}%")

        # Show image
        if show_image:
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.title(f'{diagnosis}\nConfidence: {confidence:.2%}',
                      fontsize=16, fontweight='bold', color=color, pad=20)
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        return diagnosis, confidence, prob

    except Exception as e:
        print(f" Error processing the image: {e}")
        return None, None, None


# ============================================================================
# 3. Predict on folder

def predict_folder(folder_path):
    """
    Predict on all images in a folder
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(image_extensions)]

    if len(image_files) == 0:
        print(f" No images found in: {folder_path}")
        return

    print(f"\n Found {len(image_files)} images")
    print("=" * 70)

    results = []

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        diagnosis, confidence, prob = predict_image(img_path, show_image=False)

        if diagnosis:
            results.append({
                'file': img_file,
                'diagnosis': diagnosis,
                'confidence': confidence,
                'probability': prob
            })

    # Summary
    print("\n" + "=" * 70)
    print(" Summary of results:")
    print("=" * 70)

    normal_count = sum(1 for r in results if 'Normal' in r['diagnosis'])
    pneumonia_count = len(results) - normal_count

    print(f"\n   Total images: {len(results)}")
    print(f"   🟢 Normal cases: {normal_count} ({normal_count / len(results) * 100:.1f}%)")
    print(f"   🔴 Pneumonia cases: {pneumonia_count} ({pneumonia_count / len(results) * 100:.1f}%)")

    # Detailed table
    print(f"\n{'File':<30} {'Diagnosis':<15} {'Confidence':<10}")
    print("-" * 60)
    for r in results:
        diag_short = r['diagnosis'].replace('🔴 ', '').replace('🟢 ', '')
        print(f"{r['file']:<30} {diag_short:<15} {r['confidence'] * 100:>6.2f}%")

    # Plot
    plot_folder_results(results)

    return results


# ============================================================================
# 4. Plot folder results

def plot_folder_results(results):
    """
    Plot folder statistics
    """
    normal_count = sum(1 for r in results if 'Normal' in r['diagnosis'])
    pneumonia_count = len(results) - normal_count

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].pie([normal_count, pneumonia_count],
                labels=['Normal', 'Pneumonia'],
                autopct='%1.1f%%',
                colors=['green', 'red'],
                startangle=90)
    axes[0].set_title('Distribution of Predictions', fontweight='bold')

    normal_conf = [r['confidence'] for r in results if 'Normal' in r['diagnosis']]
    pneumonia_conf = [r['confidence'] for r in results if 'Pneumonia' in r['diagnosis']]

    avg_normal = np.mean(normal_conf) if normal_conf else 0
    avg_pneumonia = np.mean(pneumonia_conf) if pneumonia_conf else 0

    axes[1].bar(['Normal', 'Pneumonia'],
                [avg_normal * 100, avg_pneumonia * 100],
                color=['green', 'red'])
    axes[1].set_ylabel('Average Confidence (%)')
    axes[1].set_title('Average Confidence by Diagnosis', fontweight='bold')
    axes[1].set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig('folder_results.png', dpi=300)
    print("\n Saved: folder_results.png")
    plt.show()


# 5. Compare two images

def compare_images(img_path1, img_path2):

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for idx, img_path in enumerate([img_path1, img_path2]):
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0][0]

        if prediction > 0.5:
            diagnosis = "Pneumonia"
            confidence = prediction
            color = 'red'
        else:
            diagnosis = "Normal"
            confidence = 1 - prediction
            color = 'green'

        axes[idx].imshow(img)
        axes[idx].set_title(f'{os.path.basename(img_path)}\n{diagnosis}: {confidence:.2%}',
                            fontweight='bold', color=color)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('comparison.png', dpi=300)
    print("\n Saved: comparison.png")
    plt.show()


# ===================
# 6. Interactive Menu
# ===================

def interactive_menu():
    while True:
        print("\n" + "=" * 70)
        print(" Choose an option:")
        print("=" * 70)
        print("1. Test a single image")
        print("2. Test a full folder")
        print("3. Compare two images")
        print("4. Exit")
        print("=" * 70)

        choice = input("\nYour choice (1-4): ").strip()

        if choice == '1':
            img_path = input("\n Enter image path: ").strip()
            if os.path.exists(img_path):
                predict_image(img_path)
            else:
                print("File not found!")

        elif choice == '2':
            folder_path = input("\n Enter folder path: ").strip()
            if os.path.exists(folder_path):
                predict_folder(folder_path)
            else:
                print(" Folder not found!")

        elif choice == '3':
            img1 = input("\n First image path: ").strip()
            img2 = input(" Second image path: ").strip()
            if os.path.exists(img1) and os.path.exists(img2):
                compare_images(img1, img2)
            else:
                print(" One of the files does not exist!")

        elif choice == '4':
            print("\n Goodbye!")
            break
        else:
            print(" Invalid choice!")


# ============================================================================
# 7. Usage Examples
# ============================================================================

if __name__ == "__main__":
    print("\n Usage examples:")
    print("=" * 70)

    print("\n 1- Test a single image:")
    print("   predict_image('path/to/your/image.jpg')")

    print("\n 2- Test a folder:")
    print("   predict_folder('path/to/your/folder')")

    print("\n 3- Compare two images:")
    print("   compare_images('image1.jpg', 'image2.jpg')")

    print("\n" + "=" * 70)

    run_menu = input("\nDo you want to launch the interactive menu? (y/n): ").strip().lower()

    if run_menu == 'y':
        interactive_menu()
    else:
        print("\n You can use the functions directly in your code")
        print("   or run the script again and choose 'y'")

        print("\n Running automatic example...")
        test_img = r'C:\Users\AliKhalid\Desktop\pneumonia_detection\chest_xray\chest_xray\test\PNEUMONIA\person25_virus_59.jpeg'
        if os.path.exists(test_img):
            predict_image(test_img)
        else:
            print(" Test image not found")
